# Primes
import tensorflow as tf
from tensorflow.python.framework import function

primes = function.Declare("primes", [("x", tf.int32)], [("ret", tf.int32)])
PrimePlus  = function.Declare("PrimePlus", [("n", tf.int32), ("i", tf.int32)], [("ret", tf.int32)])
PrimeMinus = function.Declare("PrimeMinus", [("n", tf.int32), ("i", tf.int32)], [("ret", tf.int32)])
test = function.Declare("test",[("n",tf.int32),("i",tf.int32)],[("ret",tf.bool)])

@function.Defun(tf.int32, func_name="primes", out_names=["ret"])
def PrimesImpl(n):
	return tf.cond(tf.less_equal(n, 0),
		lambda: 2,
		lambda: tf.cond(tf.equal(n, 1),
			lambda: 3,
			lambda: PrimeMinus(n-2,1)))
PrimesImpl.add_to_graph(tf.get_default_graph())

@function.Defun(tf.int32, tf.int32, func_name="PrimeMinus", out_names=["ret"])
def FindPrimeMinusImpl(n,i):
	return tf.cond(test(6*i-1, 1),
		lambda: tf.cond(tf.equal(n, 0),
			lambda: 6*i-1,
			lambda: PrimePlus(n-1,i)),
		lambda: PrimePlus(n,i))
FindPrimeMinusImpl.add_to_graph(tf.get_default_graph())

@function.Defun(tf.int32, tf.int32, func_name="PrimePlus", out_names=["ret"])
def FindPrimePlusImpl(n,i):
	return tf.cond(test(6*i-1, 1),
		lambda: tf.cond(tf.equal(n, 0),
			lambda: 6*i-1,
			lambda: PrimeMinus(n-1,i+1)),
		lambda: PrimeMinus(n,i+1))
FindPrimePlusImpl.add_to_graph(tf.get_default_graph())

@function.Defun(tf.int32, tf.int32, func_name="test", out_names=["ret"])
def TestPrimeImpl(n,i):
	return tf.cond(tf.greater((6*i-1)*(6*i-1), n),
		lambda: True,
		lambda: tf.cond(tf.equal(tf.mod(n, (6*i-1)), 0),
			lambda: False,
			lambda: tf.cond(tf.equal(tf.mod(n, (6*i-1)), 0),
				lambda: False, lambda: test(n, i+1))))
TestPrimeImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
with tf.Session() as sess:
	print(sess.run(primes(n), feed_dict={n:7500}))


