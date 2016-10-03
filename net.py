"""Convolutional neural net for Solar flare learning
"""


import tensorflow as tf
import datareader


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
						padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var, name):
	"""Attaches summaries to a tensor
	"""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.scalar_summary('stddev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)


def conv_pool_layer(input_tensor, weight_shape, stride, layer_name):
	"""Reusable code for one convolution-pooling layer
	"""
	with tf.name_scope(layer_name):
		with tf.name_scope('w_conv'):
			W_conv = weight_variable(weight_shape)
			variable_summaries(W_conv, layer_name + '/w_conv')
		with tf.name_scope('b_conv'):
			b_conv = bias_variable([weight_shape[-1]])
			variable_summaries(b_conv, layer_name + '/b_conv')
		with tf.name_scope('conv'):
			preactivate = conv2d(input_tensor, W_conv, stride) + b_conv
			tf.histogram_summary(layer_name + '/pre_activations', preactivate)
		activations = tf.nn.relu(preactivate, 'activation')
		tf.histogram_summary(layer_name + '/activations', activations)
		return max_pool_2x2(activations)
	

# width/height of input image
PIX = 256
# number of images bundled together for time series info
BUNDLE = 4
# format of the flares is a 3*10 sparse array where the first array
# indicates C,M, or X class, second is the next digit, third is next, etc


if __name__ == '__main__':
	train, test = datareader.get_data_sets()

	tf.reset_default_graph()
	sess = tf.InteractiveSession()

	x = tf.placeholder(tf.float32, shape=[None, PIX, PIX, BUNDLE])
	y_ = tf.placeholder(tf.float32, shape=[None, 4])

	conv_pool1 = conv_pool_layer(x, [8, 8, 4, 32], 4, 'conv_pool_1')
	conv_pool2 = conv_pool_layer(conv_pool1, [4, 4, 32, 64], 2, 'conv_pool_2')
	conv_pool3 = conv_pool_layer(conv_pool2, [4, 4, 64, 64], 2, 'conv_pool_3')

	# reshaped to 2x2x64 and fed into a 1024 neuron fully connected layer
	W_fc1 = weight_variable([2 * 2 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool3_flat = tf.reshape(conv_pool3, [-1, 2*2*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.scalar_summary('dropout_keep_probability', keep_prob)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 4])
	b_fc2 = bias_variable([4])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(
				y_conv + 1e-10), reduction_indices=[1]))
		tf.scalar_summary('cross entropy', cross_entropy)
	
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary('accuracy', accuracy)

	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter('logdir/train', sess.graph)
	test_writer = tf.train.SummaryWriter('logdir/test')
	sess.run(tf.initialize_all_variables())

	for i in range(10000):
		batch = train.next_batch(50)
		if (i % 100 == 0):
			summary, acc = sess.run([merged, accuracy], feed_dict={
					x: test.images(), y_: test.labels(), keep_prob: 1.0})
			output_prediction = tf.argmax(y_conv, 1).eval(feed_dict={
					x: batch[0][:1], keep_prob: 1.0})
			test_writer.add_summary(summary, i)
			print("step {}, training accuracy {:.3f}, prediction {}".format(
					i, acc, output_prediction))
		else:
			summary, _ = sess.run([merged, train_step], feed_dict={
					x: batch[0], y_: batch[1], keep_prob: 0.5})
			train_writer.add_summary(summary, i)

	print("test accuracy %g"%accuracy.eval(feed_dict={
		  x: test.images(), y_: test.labels(), keep_prob: 1.0}))
