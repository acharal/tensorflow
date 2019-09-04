/*

Licensed under the Apache License, Version 2.0 (the "License");
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
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {
namespace {


class FunctionTransformationTest : public ::testing::Test {

};

TEST_F(FunctionTransformationTest, NoTrans) {

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
  Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
  Output c = ops::AddN(s.WithOpName("c").WithDevice("/CPU:0"), {a, b});
  Output d = ops::AddN(s.WithOpName("d"), {b, c});

  GrapplerItem item;
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  FunctionTransformation func_trans;
  GraphDef output;
  Status status = func_trans.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
