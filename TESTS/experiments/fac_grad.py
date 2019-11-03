import tensorflow as tf
from tensorflow.python.framework import function

fac = function.Declare("Fac", [("n", tf.float32)], [("ret", tf.float32)])

@function.Defun(tf.float32, func_name="Fac", grad_func="GradFac", create_grad_func=True, out_names=["ret"])
def FacImpl(n):
	return tf.cond(tf.less_equal(n, 1),
		lambda: tf.constant(1.0),
		lambda: n * fac(n - 1))

FacImpl.add_to_graph(tf.get_default_graph())

n_var = tf.get_variable('n_var', [], initializer=tf.constant_initializer(4.0))
x = tf.add(n_var, 1)
res1 = fac(x)
y = tf.add(res1, 1)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(y)
#print(tf.get_default_graph().as_graph_def())
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(n_var.eval(session=sess))
print(sess.run(train_op))
print(n_var.eval(session=sess))

writer.close()

sess.close()
