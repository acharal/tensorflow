This is a brief changelog that describes the changes made to the codebase in order to
support recursive function definitions.

* Add [`function_transformation.{h,cc}`](tensorflow/core/grappler/optimizers/function_transformation.h)
  to grappler optimizers to transform a source set of dataflow graph with user-defined functions 
  to a single `Call`/`Return` dataflow graph.

* Change [`meta_optimizer.cc`](tensorflow/core/grappler/optimizers/meta_optimizer.c)
  and [`rewriter_config.proto`](tensorflow/core/protobuf/rewriter_config.proto)
  to include `function_transformation`.

* Add [`function_control_ops.{h,cc}`](tensorflow/core/ops/function_control_ops.cc) to
  defined `Call`/`Return` nodes and also their [kernels](tensorflow/core/kernels/function_control_ops.cc).

* Change [`op_types.{h,cc}`](tensorflow/core/grappler/op_types.cc) to include convenient
  checks for `Call` and `Return` usable by the grappler optimizers.

* Change [`graph.{h,cc}`](tensorflow/core/graph/graph.cc) and
  [`graph_constructor.cc`](tensorflow/core/graph/graph_constructor.cc) to
  include `Call`/`Return` nodes and check operations for them (e.g.`IsCall`, `IsReturn`)

* Change [`graph_partition.cc`](tensorflow/core/graph/graph_partition.cc) to partition dataflow 
  graphs that may contain `Call`/`Return` nodes.

* Change [`executor.cc`](tensorflow/core/common_runtime/executor.cc) to handle `Call`/`Return`
  nodes and the respective tags that these nodes generate.

* Change [`constant_folding.cc`](tensorflow/core/grappler/optimizers/constant_folding.c)
  to define that `Call`/`Return` cannot be unfolded.

* Change [`topological_sort.cc`](tensorflow/core/grappler/utils/topological_sort.cc)
  to also handle cycles produced by the `Call`/`Return` operators.
