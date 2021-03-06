import tensorflow as tf
from tensorflow.python.framework import function

fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
def FacImpl(n):
	return fac(n)


FacImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
x = tf.add(n, 1)
result = fac(x)
#y = tf.add(result, 1)

#print(tf.get_default_graph().as_graph_def())

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()
print(sess.run(result, feed_dict={n: 5}))

writer.close()

sess.close()
