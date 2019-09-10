import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl


fac = function.Declare("Func", [("n", tf.float32)], [("ret", tf.float32)])
@function.Defun(tf.float32, func_name="Func",  create_grad_func=True,  out_names=["ret"])
def FacImpl(n):
	return tf.cond(tf.less_equal(n, 1),
		lambda: tf.add(n, 2),
		lambda: tf.add(n, 1))

FacImpl.add_to_graph(tf.get_default_graph())

# return tf.cond(tf.less_equal(n, 1),
# 	lambda: tf.constant(0.0),
# 	lambda: tf.constant(1.0))
# return tf.add(n,n)


n_var = tf.get_variable('n_var', [], initializer=tf.constant_initializer(0.0))

x = tf.add(n_var, 1)
result = fac(x)
y = tf.add(result, 1)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(y)		
#print(tf.get_default_graph().as_graph_def())

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()

# sess.run(tf.initialize_all_variables())

print(sess.run(train_op))

writer.close()

sess.close()
