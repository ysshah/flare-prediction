"""Convolutional neural net for Solar flare learning
"""


import tensorflow as tf
import datareader
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


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


def nn_layer(input_tensor, weight_shape, layer_name, relu=True, stride=None):
        """Reusablee code for one nn layer
        - Bias shape determined using the last dim of the weight shape
        - Defaults to using ReLU unless 'relu' is set to false. Uses softmax then.
        - Stride used to determine if convolution is necessary. If left at none,
        a fully connected layer is created.
        """
        with tf.name_scope(layer_name):
                with tf.name_scope('weights'):
                        W = weight_variable(weight_shape)
                        variable_summaries(W, layer_name + '/weights')
                with tf.name_scope('biases'):
                        b = bias_variable(weight_shape[-1:])
                        variable_summaries(b, layer_name + '/biases')

                if stride is None:
                        with tf.name_scope('Wx_plus_b'):
                                preactivate = tf.matmul(input_tensor, W) + b
                else:
                        with tf.name_scope('conv'):
                                preactivate = conv2d(input_tensor, W, stride) + b
                tf.histogram_summary(layer_name + '/Wx_plus_b', preactivate)

                if relu:
                        activations = tf.nn.relu(preactivate, 'relu')
                else:
                        activations = tf.nn.softmax(preactivate, 'softmax')
                tf.histogram_summary(layer_name + '/activations', activations)

                if stride is not None:
                        activations = max_pool_2x2(activations)
                # tf.image_summary('activations', activations)
                return activations


def save_distribution(predictions, actual, i):
        """Saves figures showing the prediction distribution for the net
        alongside the actual distribution
        """
        p_binned = np.bincount(predictions)
        p_binned = np.pad(p_binned, (0, 4-len(p_binned)), 'constant',
                        constant_values=0)
        labels = ['no flare', 'C flare', 'M flare', 'X flare']
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.pie(p_binned, labels=labels)
        plt.title('Distribution of predictions at step {}'.format(i))

        a_binned = np.bincount(tf.argmax(actual, 1).eval())
        a_binned = np.pad(a_binned, (0, 4-len(a_binned)), 'constant',
                        constant_values=0)
        plt.subplot(122)
        plt.pie(a_binned, labels=labels)
        plt.title('Actual distribution at step {}'.format(i))
        if not os.path.exists('figs'):
                os.makedirs('figs')
        plt.savefig('figs/step{}.pdf'.format(i))
        plt.close()


def main(train, test):
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        # 256x256 is image size; 4 in number of channels bundled together
        x = tf.placeholder(tf.float32, shape=[None, 256, 256, 4],
                        name='images')
        x_image = tf.reshape(x, [-1, 256, 256, 1], 'sketch_image')
        tf.image_summary('curls', tf.reshape(x, [-1, 256, 256, 1]))
        # 4 output categories: no flare, C, M, or X
        y_ = tf.placeholder(tf.float32, shape=[None, 4], name='labels')

        conv_pool1 = nn_layer(x, [8, 8, 4, 32], 'conv_pool_1', stride=4)
        conv_pool2 = nn_layer(conv_pool1, [4, 4, 32, 64], 'conv_pool_2', stride=2)
        conv_pool3 = nn_layer(conv_pool2, [4, 4, 64, 64], 'conv_pool_3', stride=2)

        pool3_flat = tf.reshape(conv_pool3, [-1, 2*2*64])
        fc1 = nn_layer(pool3_flat, [2*2*64, 1024], 'fc_relu')

        with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                tf.scalar_summary('dropout_keep_probability', keep_prob)
                fc1_drop = tf.nn.dropout(fc1, keep_prob)
        
        y_conv = nn_layer(fc1_drop, [1024, 4], 'fc_softmax', relu=False)

        with tf.name_scope('cross_entropy'):
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(
                                y_conv + 1), reduction_indices=[1]))
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
        
        saver = tf.train.Saver()
        if os.path.exists('model.ckpt'):
                saver.restore(sess, 'model.ckpt')
                print('Sucessfully restored model parameters from save file')
        else:
                print('Unable to find/restore model parameters from save file')

        for i in range(1000000):
                batch = train.next_batch(50)
                if (i % 1000 == 0):
                        summary, acc = sess.run([merged, accuracy], feed_dict={
                                        x: test.images(), y_: test.labels(), keep_prob: 1.0})
                        test_writer.add_summary(summary, i)
                        print("step {}, training accuracy {:.3f}".format(i, acc))

                        predictions = tf.argmax(y_conv, 1).eval(feed_dict={
                                        x: batch[0], keep_prob: 1.0})
                        save_distribution(predictions, batch[1], i)
                        save_path = saver.save(sess, 'model.ckpt')

                if (i % 500 == 0):
                        summary = sess.run(merged, feed_dict={
                                        x: batch[0], y_: batch[1], keep_prob: 0.5})
                        train_writer.add_summary(summary, i)

                train_step.run(feed_dict={
                                x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g"%accuracy.eval(feed_dict={
                  x: test.images(), y_: test.labels(), keep_prob: 1.0}))


def clean():
        print('clearing logdir data, figures, and save files')
        try:
                shutil.rmtree('logdir/train/')
        except FileNotFoundError:
                pass
        try:
                shutil.rmtree('logdir/test/')
        except FileNotFoundError:
                pass
        for f in os.listdir('figs'):
                os.remove(os.path.join('figs/', f))
        try:
                os.remove('checkpoint')
        except FileNotFoundError:
                pass
        try:
                os.remove('model.ckpt')
        except FileNotFoundError:
                pass
        try:
                os.remove('model.ckpt.meta')
        except FileNotFoundError:
                pass


if __name__ == '__main__':
        import sys
        if ('--clean' in sys.argv):
            clean()
        train, test = datareader.get_data_sets()
        main(train, test)
