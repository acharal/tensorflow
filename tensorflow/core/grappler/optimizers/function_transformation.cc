/* Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/function_transformation.h"
#include <set>
#include <iostream>
#include <unordered_map>
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/util/events_writer.h"

#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"

namespace tensorflow {
namespace grappler {
namespace {

typedef std::unordered_map<string, NodeDef*> ArgMergeMap;

struct FuncInfo {
  ArgMergeMap argMergeMap;
  gtl::ArraySlice<string> fetch;

  std::vector<NodeDef*> inputs;
  std::vector<OpDef::ArgDef> input_def;
  std::vector<string> outputs;
  std::vector<OpDef::ArgDef> output_def;
};

// same with commit b691c0 (possibly)
class FunctionInliningContext {
  public:
    explicit FunctionInliningContext(const GrapplerItem& item)
            : library_(&item.graph.library()), functions_(InliningCandidates(item)) {}

    const FunctionDefLibrary& Library() const { return *library_; }

    bool HasInlinedFunctions() const { return !functions_.empty(); }

    // Find inlining candidate by name. Return nullptr if not found.
    const FunctionDef* FindInlinedFunction(const string& name) const {
      auto it = functions_.find(name);
      if (it != functions_.end()) {
        return it->second;
      } else {
        return nullptr;
      }
    }

  private:
    std::unordered_map<string, const FunctionDef*> InliningCandidates(const GrapplerItem& item) const {
      std::unordered_map<string, const FunctionDef*> functions;
      for (const FunctionDef& func : item.graph.library().function()) {
        // Don't inline functions marked as noinline
//                    if (func.attr().count("_noinline") != 0) {
//                      continue;
//                    }
        // Don't touch anything marked XLA to prevent XLA failures further down
        // the road.
        if (func.attr().count("_XlaCompile") > 0 &&
            func.attr().at("_XlaCompile").b()) {
          continue;
        }
        // Can't create IdentityN nodes with no input or output: skip these
        // functions for now.
        if (func.signature().input_arg_size() == 0 ||
            func.signature().output_arg_size() == 0) {
          continue;
        }
        functions[func.signature().name()] = &func;
      }
      return functions;
    }

    const FunctionDefLibrary* library_;
    std::unordered_map<string, const FunctionDef*> functions_;

    TF_DISALLOW_COPY_AND_ASSIGN(FunctionInliningContext);
};

// Copy input/output argument type to the type_list. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const OpDef::ArgDef& arg,
                   const std::unordered_map<string, AttrValue>& func_attr,
                   DataType* type) {
    if (arg.type() != DT_INVALID) {
      *type = arg.type();
    } else {
      auto it = func_attr.find(arg.type_attr());
      if (it == func_attr.end() || it->second.type() == DT_INVALID) {
        return errors::InvalidArgument(
                "Invalid argument ", arg.name());
      }
      *type = it->second.type();
    }
    return Status::OK();
}

// Copy input/output argument type to the type_list. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const OpDef::ArgDef& arg,
                   const std::unordered_map<string, AttrValue>& func_attr,
                   AttrValue::ListValue* type_list) {
    if (arg.type() != DT_INVALID) {
      type_list->add_type(arg.type());
    } else {
      auto it = func_attr.find(arg.type_attr());
      if (it == func_attr.end() || it->second.type() == DT_INVALID) {
        return errors::InvalidArgument("Invalid argument ", arg.name());
      }
      type_list->add_type(it->second.type());
    }
    return Status::OK();
}

struct CallInfo {
    int call_id;
    NodeDef* node;
    string node_name;
    string function_name;
    string device;
    std::vector<string> input_nodes;
    std::unordered_map<string, AttrValue> attr;
};

class CallRewriter {

  public:
    explicit CallRewriter(const GrapplerItem item_, GraphDef* graph_, const FunctionInliningContext& ctx_)
        : graph(graph_), ctx(ctx_), item(item_) { }

    ~CallRewriter() {
        Finalize();
    }

    Status CollectCalls(std::vector<CallInfo>& calls);

    Status TransformCall(CallInfo& call_info);

    // Inlines a function to item.graph and if already inlined provide func_info
    Status FindCompatibleOrInlineFunction(const string& name,
        const std::unordered_map<string, AttrValue>& func_attr,
        const string& device,
        GraphDef* optimized_graph, FuncInfo& func_info);

    void Finalize() {
        if (!nodes_to_delete.empty()) {
            // garbage collect the transformed call nodes
            int last = graph->node_size() - 1;
            for (int i = graph->node_size() - 1; i >= 0; --i) {
                const NodeDef& node = graph->node(i);
                if (nodes_to_delete.find(node.name()) != nodes_to_delete.end()) {
                    graph->mutable_node()->SwapElements(i,last);
                    last--;
                }
            }

            graph->mutable_node()->DeleteSubrange(last + 1,
                                                  graph->node_size() - last - 1);

            nodes_to_delete.clear();
        }

        if (!output_map_.empty()) {
            // change all the recorded outputs;
            // the new outputs where produced by the addition of the RetOp and
            // the substitution was deferred to increase performance
            for (NodeDef& node : *graph->mutable_node()) {
                for (string& in : *node.mutable_input()) {
                    auto it = output_map_.find(in);
                    if (it != output_map_.end()) {
                        in = it->second;
                    }
                }
            }
            output_map_.clear();
        }
    }

    inline int GetCallId(const NodeDef& node) { int call_id = id; id++; return call_id; }

  private:
    Status AddCallOp(const CallInfo& call_info, const OpDef::ArgDef arg,
                   const string& input, int arg_id, NodeDef* call_node);

    Status AddRetOp(const CallInfo& call_info, const OpDef::ArgDef arg,
                  const string& input, int arg_id, NodeDef* ret_node);

    Status ConnectInput(NodeDef* from, NodeDef* to);

    bool ShouldPreserveOutputs(const string& node) {
        for (const string& fetch_out : item.fetch) {
            if (NodeName(fetch_out) == node)
                return true;
        }
        return false;
    }

    void ReplaceOutput(const string& old_output, const string& new_output) {
        // maybe some more checks
        output_map_[old_output] = new_output;
    }

    void MarkCallTransformed(CallInfo& call_info) {
        NodeDef* node = call_info.node;
        node->clear_input();
        node->set_op("NoOp");
        node->set_name(strings::StrCat("$$$", node->name()));
        nodes_to_delete.insert(node->name());
    }

    GraphDef* graph;
    const FunctionInliningContext& ctx;
    const GrapplerItem item;
    std::unordered_map<string, FuncInfo> transformed_functions_;
    std::unordered_map<string, string> output_map_;
    std::set<string> nodes_to_delete;
    int id = 0;

    TF_DISALLOW_COPY_AND_ASSIGN(CallRewriter);
};


Status CallRewriter::CollectCalls(std::vector<CallInfo>& calls) {

    // identify and collect calls in the graph
    for (NodeDef& node : *graph->mutable_node()) {
        const FunctionDef* func = ctx.FindInlinedFunction(node.op());
        if (func != nullptr) {
            CallInfo call;
            call.call_id = GetCallId(node);
            call.node_name = node.name();
            call.function_name = node.op();
            call.node = &node;
            call.device = node.device();

            std::unordered_map<string, AttrValue> call_attr(node.attr().begin(), node.attr().end());
            call.attr = call_attr;

            int input_size = func->signature().input_arg_size();
            call.input_nodes.resize(input_size);
            for (int i = 0; i < input_size; i++) {
                call.input_nodes[i] = node.input(i);
            }
            calls.push_back(call);
        }
    }
    return Status::OK();
}

Status CallRewriter::AddCallOp(const CallInfo& call_info,
               const OpDef::ArgDef arg,
               const string& input,
               int arg_id, NodeDef* call) {
    string prefix = call_info.node_name;
    string call_name = strings::StrCat("Call", "_", arg_id);
    call->set_name(AddPrefixToNodeName(call_name, prefix));
    call->set_op("Call");
    //call->set_device(node.device());
    call->add_input(input);

    DataType type;
    TF_RETURN_IF_ERROR(CopyArgType(arg, call_info.attr, &type));

    auto& attr = *call->mutable_attr();

    //SetArgType(arg, call_info.attr, attr);

    attr["T"].set_type(type);
    attr["frame_name"].set_s(call_info.function_name);
    attr["call_id"].set_i(call_info.call_id);
    attr["arg_id"].set_i(arg_id);
    attr["is_constant"].set_b(false);

    return Status::OK();
}

Status CallRewriter::AddRetOp(const CallInfo& call_info,
              const OpDef::ArgDef arg,
              const string& input,
              int arg_id, NodeDef* ret) {
    string prefix = call_info.node_name;
    string ret_name = strings::StrCat("Ret", "_", arg_id);
    ret->set_name(AddPrefixToNodeName(ret_name, prefix));
    ret->set_op("Return");
    ret->add_input(input);

    DataType type;
    TF_RETURN_IF_ERROR(CopyArgType(arg, call_info.attr, &type));

    auto& attr = *ret->mutable_attr();
    attr["T"].set_type(type);
    attr["frame_name"].set_s(call_info.function_name);
    attr["call_id"].set_i(call_info.call_id);
    attr["arg_id"].set_i(arg_id);

    return Status::OK();
}

Status CallRewriter::ConnectInput(NodeDef* from, NodeDef* to) {
    int to_input = to->input_size();
    if (to_input == 1) {
        // it is Identity and we convert it to Merge.
        CHECK(IsIdentity(*to));
        to->set_op("Merge");
    }
    to->add_input(from->name());
    if (to->input_size() > 1) {
        (*to->mutable_attrs())["N"].set_i(to->input_size());
    }
    return Status::OK();
}

Status CallRewriter::TransformCall(CallInfo& call_info) {
    FuncInfo func_info;

    // inlines the body of a function and provides a struct with func_info
    TF_RETURN_IF_ERROR(FindCompatibleOrInlineFunction(
        call_info.function_name, call_info.attr, call_info.device, graph, func_info));

    CHECK_EQ(call_info.input_nodes.size(), func_info.inputs.size());

    std::vector<NodeDef*> call_nodes;
    std::vector<NodeDef*> ret_nodes;

    call_nodes.resize(func_info.inputs.size());
    for (unsigned int arg_num = 0; arg_num < func_info.inputs.size(); arg_num++) {
        call_nodes[arg_num] = graph->add_node();
        AddCallOp(call_info,
                func_info.input_def[arg_num],
                call_info.input_nodes[arg_num],
                arg_num,
                call_nodes[arg_num]);

        call_nodes[arg_num]->set_device(call_info.device);

        // connect the input of the inlined function to feed from call.
        TF_RETURN_IF_ERROR(ConnectInput(call_nodes[arg_num], func_info.inputs[arg_num]));
    }

    ret_nodes.resize(func_info.outputs.size());
    for (unsigned int out_port = 0; out_port < func_info.outputs.size(); out_port++) {
        ret_nodes[out_port] = graph->add_node();
        AddRetOp(call_info,
               func_info.output_def[out_port],
               func_info.outputs[out_port],
               out_port,
               ret_nodes[out_port]);
        ret_nodes[out_port]->set_device(call_info.device);
    }

    // for each call create a control dependency to each return
    // to facilitate dead propagation semantics
    for (NodeDef* ret : ret_nodes) {
        for (NodeDef* call : call_nodes)
        *(ret->add_input()) = AsControlDependency(call->name());
    }

    if (ShouldPreserveOutputs(call_info.node_name)) {
        // create an IdentityN with the same name of the initial function call
        // so as to preserve the naming of the outputs.
        // we re-use the initial node and we change (a) the op to IdentityN and
        // (b) the inputs to point to the outputs of the ret_nodes
        // The other information such as types, device placement etc remain the same.
        // The IdentityN node will sync the outputs and therefore may result to performance degradation.
        NodeDef* out = graph->add_node();
        out->set_op("IdentityN");
        out->set_name(call_info.node_name);
        out->set_device(call_info.device);
        AttrValue::ListValue* type_list = (*out->mutable_attr())["T"].mutable_list();
        for (const OpDef::ArgDef& arg : func_info.output_def) {
          TF_RETURN_IF_ERROR(CopyArgType(arg, call_info.attr, type_list));
        }
        for (unsigned int i = 0; i < func_info.outputs.size(); i++) {
            *out->add_input() = ret_nodes[i]->name();
        }
    } else {
        for (unsigned int out_port = 0; out_port < func_info.outputs.size(); out_port++) {
            ReplaceOutput(strings::StrCat(call_info.node_name, ":", out_port), ret_nodes[out_port]->name());
        }
        if (func_info.outputs.size() == 1) {
            ReplaceOutput(call_info.node_name, ret_nodes[0]->name());
        }
    }
    printf("Mark call %s (function %s) as transformed\n", call_info.node_name.c_str(), call_info.function_name.c_str());
    MarkCallTransformed(call_info);

    return Status::OK();
}

Status InlineFunction(const FunctionDef& func_def,
                      const FunctionInliningContext& ctx,
                      const std::unordered_map<string, AttrValue>& func_attr,
                      const string& device,
                      GraphDef* graph, FuncInfo& func_info) {
    std::unique_ptr<GrapplerItem> item = GrapplerItemFromFunctionDef(func_def, func_attr, ctx.Library());
    string prefix = func_def.signature().name();

    if (!item) {
        return errors::InvalidArgument(
                 "Failed to inline function ", func_def.signature().name());
    }
    int arg_size = func_def.signature().input_arg_size();
    // create an inverse map of arg to provide name -> argument number
    std::unordered_map<string, int> input_nodes;
    for (int i = 0; i < arg_size; ++i) {
        const OpDef::ArgDef& arg = func_def.signature().input_arg(i);
        input_nodes[arg.name()] = i;
    }
    func_info.inputs.resize(arg_size);
    func_info.input_def.resize(arg_size);
    for (int i = 0; i < arg_size; ++i) {
        const OpDef::ArgDef& arg = func_def.signature().input_arg(i);
        NodeDef* merge = graph->add_node();
        merge->set_name(AddPrefixToNodeName(strings::StrCat("Input", "_", i), prefix));
        merge->set_op("Identity");
        merge->set_device(device);

        DataType type;
        TF_RETURN_IF_ERROR(CopyArgType(arg, func_attr, &type));
        auto& attr = *merge->mutable_attr();
        attr["T"].set_type(type);

        func_info.inputs[i] = merge;
        func_info.input_def[i] = arg;
    }

    // prefix each node in function graph and place it to the global graph.
    // the inputs of each node need to be renamed as well to reflect the change.
    for (NodeDef& func_body_node : *item->graph.mutable_node()) {
        const string& curr_name = func_body_node.name();
        // If the func body node is func's input argument
        auto input_it = input_nodes.find(curr_name);

        if (input_it != input_nodes.end()) {
            CHECK_EQ(0, func_body_node.input_size());
            // Turn input placeholders into identity nodes
            if (IsPlaceholder(func_body_node)) {
                func_body_node.set_op("Identity");
            }
            // Connect merge with input arg
            func_body_node.add_input(func_info.inputs[input_it->second]->name());
        } else {
            // Else if not an input_arg_node
            // Update the input names if any.
            for (string& input : *func_body_node.mutable_input()) {
                input = AddPrefixToNodeName(input, prefix);
            }
            // If the node has no input, make hook it up to the Merge nodes to ensure
            // it runs in the same frame as the other nodes of the function body.
            if (func_body_node.input_size() == 0) {
                for (auto& func_input_node : func_info.inputs) {
                 *func_body_node.add_input() = AsControlDependency(func_input_node->name());
                }
            }
        }

        // Add the node name as a prefix to avoid collisions after inlining
        func_body_node.set_name(AddPrefixToNodeName(curr_name, prefix));

        // Make sure the node is placed
        if (func_body_node.device().empty())
          func_body_node.set_device(device);

        // Move the node to the main graph
        graph->add_node()->Swap(&func_body_node);
    }

    func_info.outputs.clear();
    func_info.outputs.resize(item->fetch.size());
    func_info.output_def.resize(item->fetch.size());

    for (unsigned int i = 0; i < item->fetch.size(); i++) {
        func_info.outputs[i] = AddPrefixToNodeName(item->fetch[i], prefix);
        func_info.output_def[i] = func_def.signature().output_arg(i);
    }

    return Status::OK();
}

// new
Status CallRewriter::FindCompatibleOrInlineFunction(
            const string& func_name,
            const std::unordered_map<string, AttrValue>& func_attr,
            const string& device,
            GraphDef* graph,
            FuncInfo& func_info) {
    const auto& it = transformed_functions_.find(func_name);
    // maybe it is not wise to discard call attributes
    // possible type specialization?
    if (it != transformed_functions_.end()) {
        func_info = it->second;
        return Status::OK();
    }
    const FunctionDef* func_def = ctx.FindInlinedFunction(func_name);
    if (func_def == nullptr) {
        return errors::InvalidArgument(
                        "Invalid argument, function ", func_name, "can not be found",
                        "or not marked to be inlined");
    }
    TF_RETURN_IF_ERROR(
        InlineFunction(*func_def, ctx, func_attr, device, graph, func_info));
    transformed_functions_[func_name] = func_info;
    printf("Store inlined function %s\n", func_name.c_str());
    return Status::OK();
}

}  // namespace

Status FunctionTransformation::Optimize(Cluster* cluster, const GrapplerItem& item,
                                        GraphDef* output) {
    FunctionInliningContext ctx(item);
    CallRewriter call_rewriter(item, output, ctx);

    *output = item.graph;
    if (!ctx.HasInlinedFunctions()) {
        return Status::OK();
    }

    std::vector<CallInfo> calls;
    while (1) {
        TF_RETURN_IF_ERROR(call_rewriter.CollectCalls(calls));
        if (calls.empty()) {
            break;
        }
        for (CallInfo& call : calls) {
            Status s = call_rewriter.TransformCall(call);
            if (!s.ok()) {
              printf("Error: %s\n", s.error_message().c_str());
              return s;
            }
            printf("After transforming call %s:\n %s\n", call.function_name.c_str(), SummarizeGraphDef(*output).c_str());
        }
        calls.clear();
    }
    call_rewriter.Finalize();
    printf("After finalizing:\n %s\n", SummarizeGraphDef(*output).c_str());
    *output->mutable_versions() = item.graph.versions();

    // Function Library should be pruned of unreachable function definitions
    // cf. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/function_optimizer.cc#L428
    // however in this version there is a check in meta_optimizer that guarantees
    // that function library remains of the same length
    // cf. https://github.com/acharal/tensorflow/blob/r1.4_recursion/tensorflow/core/grappler/optimizers/meta_optimizer.cc#L132
    *output->mutable_library() = item.graph.library();



    /******************************************************************************************************/
    // Dumps optimized graph in a not so readable form
    // const GraphDef* tmp = optimized_graph;
    // printf("Summarize Optimized Graph\n %s\n", SummarizeGraphDef(*tmp).c_str());
    // Write an event, so that we can visualize this optimized graph in tensorboard
    EventsWriter writer("TRANSFORMATION");
    Event event;
    event.set_wall_time(1234);
    event.set_step(34);
    const size_t proto_size = output->ByteSizeLong();
    void* buf = port::Malloc(proto_size);
    if (buf == nullptr) {
    return errors::ResourceExhausted(
              "Failed to allocate memory to serialize message of type '" ,
              output->GetTypeName(), "' and size ", proto_size);
    }
  output->SerializeToArray(buf, proto_size);
    const void* bf = buf;
    event.set_graph_def(bf, proto_size);
    writer.WriteEvent(event);
    /******************************************************************************************************/

    return Status::OK();
}

void FunctionTransformation::Feedback(Cluster* cluster, const GrapplerItem& item,
                                      const GraphDef& optimized_graph,
                                      double result) {
    // Nothing to do for FunctionTransformation.
}

}  // end namespace grappler
}  // end namespace tensorflow
