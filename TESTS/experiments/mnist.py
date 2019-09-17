import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 1
batch_size = 10

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
# W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
# b1 = tf.Variable(tf.random_normal([300]), name='b1')
W1 = tf.get_variable('W1', [784, 300], initializer=tf.constant_initializer(0.0))
b1 = tf.get_variable('b1', [300], initializer=tf.constant_initializer(0.0))

# and the weights connecting the hidden layer to the output layer
# W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
# b2 = tf.Variable(tf.random_normal([10]), name='b2')

W2 = tf.get_variable('W2', [300, 10], initializer=tf.constant_initializer(0.0))
b2 = tf.get_variable('b2', [10], initializer=tf.constant_initializer(0.0))

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
# init_op = tf.global_variables_initializer()
init_op = tf.initialize_all_variables()
# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

file = open("mnist_output.txt", "w") 

# start the session
with tf.Session() as sess:
	# initialise the variables
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
	
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
	
	file.write("Epoch:" + str(epoch+1) + " cost =" + str(avg_cost) + "\n")
	file.write(str((sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))) + "\n")

file.close()