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
#include "tensorflow/core/framework/node_def_util.h"
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

static constexpr const char* const kCallOp = "Call";
static constexpr const char* const kRetOp = "Return";
static constexpr const char* const kIdentityOp = "Identity";
static constexpr const char* const kIdentityNOp = "IdentityN";
static constexpr const char* const kMergeOp = "Merge";
static constexpr const char* const kGradientOp =
    FunctionLibraryDefinition::kGradientOp;
static constexpr const char* const kFuncAttr =
    FunctionLibraryDefinition::kFuncAttr;

struct FuncInfo { 
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  std::vector<NodeDef*> args;
  std::vector<string> rets;
};

struct FuncGradInfo {
  FuncInfo f;
  FuncInfo g;
};

// same with commit b691c0 (possibly)
class FunctionInliningContext {
  public:
    explicit FunctionInliningContext(const GrapplerItem& item)
            : library_(&item.graph.library()), functions_(InliningCandidates(item)) {

      // Todo: Deallocate this afterwards
      lib_def_ = new FunctionLibraryDefinition(OpRegistry::Global(), item.graph.library());

      get_func_sig_ = [this](const string& op, const OpDef** sig) {
          return lib_def_->LookUpOpDef(op, sig);
      };
    }

    const FunctionDefLibrary& Library() const { return *library_; }
    const FunctionLibraryDefinition* Libdef() const { return lib_def_; }
    const std::function<Status(const string&, const OpDef**)> GetFuncSig() const {return get_func_sig_;}

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

    const FunctionDef* FindInlinedFunctionAndGradient(const string& name) const {
      string grad_name = strings::StrCat(name, "Grad");
      return FindInlinedFunction(grad_name);
    }

  private:
    std::unordered_map<string, const FunctionDef*> InliningCandidates(const GrapplerItem& item) const {
      std::unordered_map<string, const FunctionDef*> functions;
      for (const FunctionDef& func : item.graph.library().function()) {

        // Don't inline functions marked as noinline
        // if (func.attr().count("_noinline") != 0) {
        //   continue;
        // }
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
    const FunctionLibraryDefinition* lib_def_;
    std::unordered_map<string, const FunctionDef*> functions_;
    std::function<Status(const string&, const OpDef**)> get_func_sig_;

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
    string call_frame;
    NodeDef* fcall = nullptr;
    NodeDef* gcall = nullptr;
    bool hasGradient() const { return (gcall != nullptr); }
};

class CallRewriter {

  public:
    explicit CallRewriter(const GrapplerItem item_, GraphDef* graph_, const FunctionInliningContext& ctx_)
        : graph(graph_), ctx(ctx_), item(item_) { }

    ~CallRewriter() {
        Flush();
    }

    Status CollectCalls(std::vector<CallInfo>& calls);

    Status TransformCall(CallInfo& call_info);

    // Inlines a function to item.graph and if already inlined provide func_info
    Status FindCompatibleOrInlineFunction(
        const CallInfo& call,
        GraphDef* optimized_graph,
        FuncGradInfo& func_info);

    void Flush() {
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
    Status AddCallOp(const CallInfo& call_info, const DataType& type,
                   const string& input, const string& prefix, int arg_id, NodeDef* call_node);

    Status AddRetOp(const CallInfo& call_info, const DataType& type,
                  const string& input, const string& prefix, int arg_id, NodeDef* ret_node);

    Status ConnectInput(NodeDef* from, NodeDef* to);

    Status TransformNode(CallInfo& info, 
            NodeDef* call, const FuncInfo& f, 
            std::vector<NodeDef*>& call_nodes,
            std::vector<NodeDef*>& ret_nodes);

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
      MarkNodeDelete(call_info.fcall);
      MarkNodeDelete(call_info.gcall);
    }

    void MarkNodeDelete(NodeDef* n) {
      n->clear_input();
      n->set_op("NoOp");
      n->set_name(AddPrefixToNodeName(n->name(), "$MarkToDelete$"));
      nodes_to_delete.insert(n->name());
    }

    GraphDef* graph;
    const FunctionInliningContext& ctx;
    const GrapplerItem item;
    std::unordered_map<string, FuncGradInfo> transformed_functions_;
    std::unordered_map<string, string> output_map_;
    std::set<string> nodes_to_delete;
    int id = 0;

    TF_DISALLOW_COPY_AND_ASSIGN(CallRewriter);
};


Status CallRewriter::CollectCalls(std::vector<CallInfo>& calls) {

    std::unordered_map<string,CallInfo> call_map;
    std::vector<NodeDef*> gradients;

    // identify and collect calls in the graph
    for (NodeDef& node : *graph->mutable_node()) {
        if (node.op() == kGradientOp) {
            gradients.push_back(&node);
        } else {
            const FunctionDef* func_def = ctx.FindInlinedFunction(node.op());
            if (func_def != nullptr) {
                CallInfo& call = call_map[node.name()];
                call.call_id = GetCallId(node);
                call.call_frame = node.op();
                call.fcall  = &node;
            }
        }
    }
    for (NodeDef* gcall : gradients) {
        const string& n = gcall->attr().at("_n").s();
        auto fcall_it = call_map.find(n);
        if (fcall_it == call_map.end()) {
            return errors::InvalidArgument("Cannot find forward node for gradient ",
                    gcall->name());
        }
        CallInfo& call = fcall_it->second;
        call.gcall = gcall;
    }

    for (const auto& it : call_map) {
        calls.push_back(it.second);
    }
    return Status::OK();
}

Status CallRewriter::AddCallOp(const CallInfo& call_info,
               const DataType& type,
               const string& input,
               const string& prefix,
               int arg_id, NodeDef* call) {
    string call_name = strings::StrCat("Call", "_", arg_id);
    call->set_op(kCallOp);
    call->set_name(AddPrefixToNodeName(call_name, prefix));
    //call->set_device(node.device());
    call->add_input(input);

    auto& attr = *call->mutable_attr();
    attr["T"].set_type(type);
    attr["frame_name"].set_s(call_info.call_frame);
    attr["call_id"].set_i(call_info.call_id);
    attr["arg_id"].set_i(arg_id);
    attr["is_constant"].set_b(false);

    return Status::OK();
}

Status CallRewriter::AddRetOp(const CallInfo& call_info,
              const DataType& type,
              const string& input,
              const string& prefix,
              int arg_id, NodeDef* ret) {
    string ret_name = strings::StrCat("Ret", "_", arg_id);
    ret->set_op(kRetOp);
    ret->set_name(AddPrefixToNodeName(ret_name, prefix));
    ret->add_input(input);

    auto& attr = *ret->mutable_attr();
    attr["T"].set_type(type);
    attr["frame_name"].set_s(call_info.call_frame);
    attr["call_id"].set_i(call_info.call_id);
    attr["arg_id"].set_i(arg_id);

    return Status::OK();
}

Status CallRewriter::ConnectInput(NodeDef* from, NodeDef* to) {
    int to_input = to->input_size();
    if (to_input == 1) {
        // it is Identity and we convert it to Merge.
        CHECK(IsIdentity(*to));
        to->set_op(kMergeOp);
    }
    to->add_input(from->name());
    if (to->input_size() > 1) {
        (*to->mutable_attr())["N"].set_i(to->input_size());
    }
    return Status::OK();
}

Status CallRewriter::TransformNode(CallInfo& info, 
        NodeDef* call, const FuncInfo& f, 
        std::vector<NodeDef*>& call_nodes,
        std::vector<NodeDef*>& ret_nodes) {
  CHECK_EQ(call->input_size(), f.args.size());

  call_nodes.resize(f.args.size());
  for (unsigned int i = 0; i < f.args.size(); i++) {
      /* check if call node is already in place, if so, validate and skip */
      if (call_nodes[i] != nullptr) {
        // TODO: validate call_id
        // TODO: validate input
        //CHECK_EQ(call_nodes[i]->input(0), call->input(i));
      } else {
        call_nodes[i] = graph->add_node();
        AddCallOp(info,
                f.arg_types[i],
                call->input(i),
                call->name(),
                i,
                call_nodes[i]);

        call_nodes[i]->set_device(call->device());

        // connect the input of the inlined function to feed from call.
        TF_RETURN_IF_ERROR(ConnectInput(call_nodes[i], f.args[i]));
      }
  }

  ret_nodes.resize(f.rets.size());
  for (unsigned int i = 0; i < f.rets.size(); i++) {
      if (ret_nodes[i] != nullptr) {
        // TODO: validate call_id
        // CHECK_EQ(ret_nodes[i]->input(0), f.rets[i]);
      } else {
        ret_nodes[i] = graph->add_node();
        AddRetOp(info,
                f.ret_types[i],
                f.rets[i],
                call->name(),
                i,
                ret_nodes[i]);
        ret_nodes[i]->set_device(call->device());
      }
  }

  // for each call create a control dependency to each return
  // to facilitate dead propagation semantics
  for (NodeDef* ret : ret_nodes) {
      for (NodeDef* call : call_nodes)
        // TODO: Check if there is already a control dependency.
        *(ret->add_input()) = AsControlDependency(call->name());
  }

  if (ShouldPreserveOutputs(call->name())) {
      // create an IdentityN with the same name of the initial function call
      // so as to preserve the naming of the outputs.
      // we re-use the initial node and we change (a) the op to IdentityN and
      // (b) the inputs to point to the outputs of the ret_nodes
      // The other information such as types, device placement etc remain the same.
      // The IdentityN node will sync the outputs and therefore may result to performance degradation.
      NodeDef* out = graph->add_node();
      out->set_op(kIdentityNOp);
      out->set_name(call->name());
      out->set_device(call->device());
      AttrValue::ListValue* type_list = (*out->mutable_attr())["T"].mutable_list();
      for (const DataType& type : f.ret_types) {
        type_list->add_type(type);
      }
      for (unsigned int i = 0; i < f.rets.size(); i++) {
          *out->add_input() = ret_nodes[i]->name();
      }
  } else {
      for (unsigned int i = 0; i < f.rets.size(); i++) {
          ReplaceOutput(strings::StrCat(call->name(), ":", i), ret_nodes[i]->name());
      }
      if (f.rets.size() == 1) {
          ReplaceOutput(call->name(), ret_nodes[0]->name());
      }
  }
  return Status::OK();
}

Status CallRewriter::TransformCall(CallInfo& call_info) {
    FuncGradInfo func_info;

    // inlines the body of a function and provides a struct with func_info
    TF_RETURN_IF_ERROR(FindCompatibleOrInlineFunction(call_info, graph, func_info));

    std::vector<NodeDef*> call_nodes;
    std::vector<NodeDef*> ret_nodes;
    std::vector<NodeDef*> gret_nodes;
    TF_RETURN_IF_ERROR(TransformNode(call_info, call_info.fcall, func_info.f, call_nodes, ret_nodes));

    if (call_info.hasGradient()) {
      // keep all the inputs of the function
      TF_RETURN_IF_ERROR(TransformNode(call_info, call_info.gcall, func_info.g, call_nodes, gret_nodes));
    }

    printf("Mark call %s (function %s) as transformed\n", call_info.fcall->name().c_str(), call_info.fcall->op().c_str());
    MarkCallTransformed(call_info);

    return Status::OK();
}

Status InlineFunction(const FunctionDef& func_def,
                      const std::unordered_map<string, AttrValue>& func_attr,
                      const FunctionInliningContext& ctx,
                      const string& device,
                      GraphDef* graph, FuncGradInfo& func_info) {
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
    func_info.f.args.resize(arg_size);
    func_info.f.arg_types.resize(arg_size);
    for (int i = 0; i < arg_size; ++i) {
        const OpDef::ArgDef& arg = func_def.signature().input_arg(i);
        NodeDef* merge = graph->add_node();
        merge->set_name(AddPrefixToNodeName(strings::StrCat("Input", "_", i), prefix));
        merge->set_op(kIdentityOp);
        merge->set_device(device);

        DataType type;
        TF_RETURN_IF_ERROR(CopyArgType(arg, func_attr, &type));
        auto& attr = *merge->mutable_attr();
        attr["T"].set_type(type);

        func_info.f.args[i] = merge;
        func_info.f.arg_types[i] = type;
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
                func_body_node.set_op(kIdentityOp);
            }
            // Connect merge with input arg
            func_body_node.add_input(func_info.f.args[input_it->second]->name());
        } else {
            // Else if not an input_arg_node
            // Update the input names if any.
            for (string& input : *func_body_node.mutable_input()) {
                input = AddPrefixToNodeName(input, prefix);
            }
            // If the node has no input, make hook it up to the Merge nodes to ensure
            // it runs in the same frame as the other nodes of the function body.
            if (func_body_node.input_size() == 0) {
                for (auto& func_input_node : func_info.f.args) {
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

    func_info.f.rets.clear();
    func_info.f.rets.resize(item->fetch.size());
    func_info.f.ret_types.resize(item->fetch.size());

    for (unsigned int i = 0; i < item->fetch.size(); i++) {
        func_info.f.rets[i] = AddPrefixToNodeName(item->fetch[i], prefix);
        DataType type;
        TF_RETURN_IF_ERROR(CopyArgType(func_def.signature().output_arg(i), func_attr, &type));
        func_info.f.ret_types[i] = type;
    }

    return Status::OK();
}

// Various helpers Print(proto) to print relevant protos to ascii.
string Print(const OpDef::ArgDef& arg) {
  string out;
  strings::StrAppend(&out, arg.name(), ":");
  if (arg.is_ref()) strings::StrAppend(&out, "Ref(");
  if (!arg.number_attr().empty()) {
    strings::StrAppend(&out, arg.number_attr(), "*");
  }
  if (arg.type() != DT_INVALID) {
    strings::StrAppend(&out, DataTypeString(arg.type()));
  } else {
    strings::StrAppend(&out, arg.type_attr());
  }
  if (arg.is_ref()) strings::StrAppend(&out, ")");
  return out;
}

// TODO(josh11b): Merge this with SummarizeAttrValue().
string Print(const AttrValue& attr_value) {
  if (attr_value.value_case() == AttrValue::kType) {
    return DataTypeString(attr_value.type());
  } else if ((attr_value.value_case() == AttrValue::kList) &&
             (attr_value.list().type_size() > 0)) {
    string ret = "{";
    for (int i = 0; i < attr_value.list().type_size(); ++i) {
      if (i > 0) strings::StrAppend(&ret, ", ");
      strings::StrAppend(&ret, DataTypeString(attr_value.list().type(i)));
    }
    strings::StrAppend(&ret, "}");
    return ret;
  } else if (attr_value.value_case() == AttrValue::kFunc) {
    if (attr_value.func().attr_size() == 0) {
      return attr_value.func().name();
    }
    std::vector<string> entries;
    for (auto p : attr_value.func().attr()) {
      entries.push_back(strings::StrCat(p.first, "=", Print(p.second)));
    }
    std::sort(entries.begin(), entries.end());
    return strings::StrCat(attr_value.func().name(), "[",
                           str_util::Join(entries, ", "), "]");
  }
  return SummarizeAttrValue(attr_value);
}

// TODO(josh11b): Merge this with SummarizeNodeDef().
string Print(const NodeDef& n) {
  string out;
  strings::StrAppend(&out, n.name(), " = ", n.op());
  if (n.attr_size() > 0) {
    std::vector<string> entries;
    for (auto& a : n.attr()) {
      entries.push_back(strings::StrCat(a.first, "=", Print(a.second)));
    }
    std::sort(entries.begin(), entries.end());
    strings::StrAppend(&out, "[", str_util::Join(entries, ", "), "]");
  }
  strings::StrAppend(&out, "(");
  std::vector<StringPiece> dat;
  std::vector<string> dep;
  for (StringPiece s : n.input()) {
    if (s.Consume("^")) {
      dep.push_back(s.ToString());
    } else {
      dat.push_back(s);
    }
  }
  strings::StrAppend(&out, str_util::Join(dat, ", "), ")");
  if (!dep.empty()) {
    strings::StrAppend(&out, " @ ", str_util::Join(dep, ", "));
  }
  return out;
}

string Print(const FunctionDef& fdef) {
  string out;
  const OpDef& sig = fdef.signature();
  strings::StrAppend(&out, "\n", sig.name());
  if (sig.attr_size() > 0) {
    strings::StrAppend(&out, "[");
    for (int i = 0; i < sig.attr_size(); ++i) {
      const auto& a = sig.attr(i);
      if (i > 0) strings::StrAppend(&out, ", ");
      if (a.type() == "type") {
        strings::StrAppend(&out, a.name(), ":", Print(a.allowed_values()));
      } else {
        strings::StrAppend(&out, a.name(), ":", a.type());
      }
    }
    strings::StrAppend(&out, "]");
  }
  strings::StrAppend(&out, "(");
  for (int i = 0; i < sig.input_arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, Print(sig.input_arg(i)));
  }
  strings::StrAppend(&out, ") -> (");
  for (int i = 0; i < sig.output_arg_size(); ++i) {
    if (i > 0) strings::StrAppend(&out, ", ");
    strings::StrAppend(&out, Print(sig.output_arg(i)));
  }
  strings::StrAppend(&out, ") {\n");
  for (const auto& n : fdef.node_def()) {
    strings::StrAppend(&out, "  ", Print(n), "\n");
  }
  for (const auto& r : fdef.ret()) {
    strings::StrAppend(&out, "  return ", r.first, " = ", r.second, "\n");
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

string Print(gtl::ArraySlice<const NodeDef*> nodes) {
  std::vector<const NodeDef*> arg;
  std::vector<const NodeDef*> ret;
  std::vector<const NodeDef*> body;
  for (const NodeDef* n : nodes) {
    if (n->op() == "_Arg") {
      arg.push_back(n);
    } else if (n->op() == "_Retval") {
      ret.push_back(n);
    } else {
      body.push_back(n);
    }
  }
  auto comp = [](const NodeDef* x, const NodeDef* y) {
    int xi;
    TF_CHECK_OK(GetNodeAttr(*x, "index", &xi));
    int yi;
    TF_CHECK_OK(GetNodeAttr(*y, "index", &yi));
    return xi < yi;
  };
  std::sort(arg.begin(), arg.end(), comp);
  std::sort(ret.begin(), ret.end(), comp);
  string out;
  strings::StrAppend(&out, "\n(");
  auto get_type = [](const NodeDef& n) {
    DataType dt;
    if (!GetNodeAttr(n, "T", &dt).ok()) {
      dt = DT_INVALID;
    }
    return DataTypeString(dt);
  };
  for (size_t i = 0; i < arg.size(); ++i) {
    const NodeDef* n = arg[i];
    if (i > 0) strings::StrAppend(&out, ", ");
    CHECK_GE(n->attr_size(), 2);
    strings::StrAppend(&out, n->name(), ":", get_type(*n));
  }
  strings::StrAppend(&out, ") -> (");
  for (size_t i = 0; i < ret.size(); ++i) {
    const NodeDef* n = ret[i];
    if (i > 0) strings::StrAppend(&out, ", ");
    CHECK_LE(2, n->attr_size());
    CHECK_EQ(1, n->input_size());
    strings::StrAppend(&out, n->input(0), ":", get_type(*n));
  }
  strings::StrAppend(&out, ") {\n");
  for (size_t i = 0; i < body.size(); ++i) {
    strings::StrAppend(&out, "  ", Print(*body[i]), "\n");
  }
  strings::StrAppend(&out, "}\n");
  return out;
}

Status InlineFunctionAndGradient(const FunctionDef* fdef,
                      const std::unordered_map<string, AttrValue>& func_attr,
                      const FunctionInliningContext& ctx,
                      const string& device,
                      GraphDef* graph, 
                      FuncGradInfo& func_info) {

    // Get func_def's gradient graph
    const FunctionDef* fgdef = ctx.FindInlinedFunctionAndGradient(fdef->signature().name());
    if (fgdef == nullptr) {
        return errors::InvalidArgument(
                "Invalid argument, function ", fgdef->signature().name(), "can not be found",
                "or not marked to be inlined");
    }

    std::unique_ptr<GrapplerItem> item = GrapplerItemFromFunctionDef(*fgdef, func_attr, ctx.Library());
    
    if (!item) {
        return errors::InvalidArgument(
                 "Failed to inline function (and its gradient)", fdef->signature().name());
    }
    string prefix = fdef->signature().name();
    size_t farg_size = fdef->signature().input_arg_size();
    size_t fret_size = fdef->signature().output_arg_size();
    size_t garg_size = fgdef->signature().input_arg_size() - farg_size;
    size_t gret_size = fgdef->signature().output_arg_size() - fret_size;

    CHECK_EQ(farg_size, gret_size);
    CHECK_EQ(garg_size, fret_size);

    func_info.f.arg_types.resize(farg_size);
    func_info.g.arg_types.resize(farg_size + garg_size);
    func_info.g.ret_types.resize(farg_size);
    for (int i = 0; i < farg_size; i++) {
      const OpDef::ArgDef& arg  = fgdef->signature().input_arg(i);
      const OpDef::ArgDef& darg = fgdef->signature().output_arg(fret_size + i);
      DataType type;
      TF_RETURN_IF_ERROR(CopyArgType(arg, func_attr, &type));
      //DataType dtype;
      //TF_RETURN_IF_ERROR(CopyArgType(darg, func_attr, &dtype));
      //CHECK_EQ(type, dtype);
      func_info.f.arg_types[i] = type;
      func_info.g.arg_types[i] = type;
      func_info.g.ret_types[i] = type;
    }


    func_info.f.ret_types.resize(fret_size);
    for (int i = 0; i < fret_size; i++) {
      const OpDef::ArgDef& arg  = fgdef->signature().output_arg(i);
      const OpDef::ArgDef& darg = fgdef->signature().input_arg(farg_size + i);
      DataType type;
      TF_RETURN_IF_ERROR(CopyArgType(arg, func_attr, &type));
      //DataType dtype;
      //TF_RETURN_IF_ERROR(CopyArgType(darg, func_attr, &dtype));
      //CHECK_EQ(type, dtype);
      func_info.f.ret_types[i] = type;
      func_info.g.arg_types[farg_size + i] = type;
    }

    // create an inverse map of arg to provide name -> argument number
    std::unordered_map<string, int> input_map;
    std::vector<string> input_names;
    input_names.resize(farg_size);
    for (int i = 0; i < farg_size + garg_size; ++i) {
        const OpDef::ArgDef& arg = fgdef->signature().input_arg(i);
        input_map[arg.name()] = i;
        if (i < farg_size) {
          input_names[i] = arg.name();
        }
    }

    func_info.f.args.resize(farg_size);
    func_info.f.rets.resize(fret_size);
    func_info.g.args.resize(farg_size + garg_size);
    func_info.g.rets.resize(gret_size);

    // prefix each node in function graph and place it to the global graph.
    // the inputs of each node need to be renamed as well to reflect the change.
    for (NodeDef& n : *item->graph.mutable_node()) {
        // If the func body node is func's input argument
        auto input_it = input_map.find(n.name());
        bool is_input = input_it != input_map.end();

        if (is_input) {
          CHECK_EQ(0, n.input_size());
          if (IsPlaceholder(n)) {
            n.set_op(kIdentityOp);
          }
        }

        // Add the node name as a prefix to avoid collisions after inlining
        n.set_name(AddPrefixToNodeName(n.name(), prefix));
        // Update the input names if any.
        for (string& input : *n.mutable_input()) {
            input = AddPrefixToNodeName(input, prefix);
        }

        // Make sure the node is placed
        if (n.device().empty())
          n.set_device(device);

        if (n.op() == kGradientOp) {
          auto& attr = *n.mutable_attr();
          auto& n_ = attr["_n"].s();
          attr["_n"].set_s(AddPrefixToNodeName(n_, prefix));
        }

        // If the node has no input, make hook it up to the Merge nodes to ensure
        // it runs in the same frame as the other nodes of the function body.
        if (!is_input && n.input_size() == 0) {
          // CHECK: constants from both in function and gradient are connected 
          // with the inputs of the function only.
          for (const string& arg : input_names) {
            *n.add_input() = AsControlDependency(AddPrefixToNodeName(arg, prefix));
          }
        }

        // Move the node to the main graph
        NodeDef* nn = graph->add_node();
        nn->Swap(&n);
        
        if (is_input) {
          int i = input_it->second;
          if (i < farg_size) {
            func_info.f.args[i] = nn;
            func_info.g.args[i] = func_info.f.args[i];
          } else { 
            func_info.g.args[i] = nn;
          }
        }
    }

    CHECK_EQ(fret_size + gret_size, item->fetch.size());

    for (unsigned int i = 0; i < fret_size + gret_size; i++) {
        string output_port = AddPrefixToNodeName(item->fetch[i], prefix);
        if (i < fret_size) {
          func_info.f.rets[i] = output_port;
        } else {
          func_info.g.rets[i - fret_size] = output_port;
        }
    }

    return Status::OK();
}

// new
Status CallRewriter::FindCompatibleOrInlineFunction(
            const CallInfo& call,
            GraphDef* graph,
            FuncGradInfo& func_info) {
    const string& func_name = call.fcall->op();
    string device = call.fcall->device();
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

    std::unordered_map<string, AttrValue> func_attr(call.fcall->attr().begin(), call.fcall->attr().end());

    if (call.hasGradient()) {
      TF_RETURN_IF_ERROR(
              InlineFunctionAndGradient(func_def, func_attr, ctx, device, graph, func_info));
    } else { 
      TF_RETURN_IF_ERROR(
              InlineFunction(*func_def, func_attr, ctx, device, graph, func_info));
    }
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
            printf("After transforming call %s:\n %s\n", call.fcall->name().c_str(), SummarizeGraphDef(*output).c_str());
        }
        calls.clear();
        call_rewriter.Flush();
    }
    call_rewriter.Flush();
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
