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
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')


def variable_summaries(var, name):
    """Attaches summaries to a tensor"""
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
    """Reusable code for one nn layer
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


def main(train, test, wind_speed=False):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # pix*pix is image size; channels is number of channels bundled together
    pix = 256
    channels = 8 if wind_speed else 4

    x = tf.placeholder(tf.float32, shape=[None, pix, pix, channels],
            name='images')
    x_image = tf.reshape(x, [-1, pix, pix, 1], 'sketch_image')
    tf.image_summary('curls', tf.reshape(x, [-1, pix, pix, 1]))
    # 4 output categories: no flare, C, M, or X for flares
    # or magnetic field, kp index, sunspot number, and dst index for wind speed
    y_ = tf.placeholder(tf.float32, shape=[None, 4], name='labels')

    if wind_speed:
        conv_pool1 = nn_layer(x, [8, 8, 8, 32], 'conv_pool_1', stride=4)
    else:
        conv_pool1 = nn_layer(x, [8, 8, 4, 32], 'conv_pool_1', stride=4)
    conv_pool2 = nn_layer(conv_pool1, [4, 4, 32, 64], 'conv_pool_2', stride=2)
    conv_pool3 = nn_layer(conv_pool2, [4, 4, 64, 64], 'conv_pool_3', stride=2)

    pool3_flat = tf.reshape(conv_pool3, [-1, 2*2*64])
    fc1 = nn_layer(pool3_flat, [2*2*64, 64], 'fc_relu')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    y_conv = nn_layer(fc1_drop, [64, 4], 'fc_softmax', relu=False)

    if wind_speed:
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(y_conv - y_))
            tf.scalar_summary('loss', loss)
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        with tf.name_scope('accuracy'):
            with tf.name_scope('corrent_prediction'):
                correct_prediction = tf.div(tf.sub(y_, y_conv), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)
    else:
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(
                    y_conv + 1), reduction_indices=[1]))
            tf.scalar_summary('cross entropy', cross_entropy)
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
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
        print('Unable to find and restore model parameters from save file')

    for i in range(1000000):
        batch = train.next_batch(50)
        if (i % 100 == 0):
            summary, acc = sess.run([merged, accuracy], feed_dict={
                    x: test.images(), y_: test.labels(), keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            t_dict = {x: batch[0][:1], keep_prob: 1.0}
            print("step\t{}, training accuracy {:.3f}, y_conv {}".format(i, acc, y_conv.eval(t_dict)))
            print('conv1\t{}'.format(quick_dist(conv_pool1.eval(feed_dict=t_dict))))
            print('conv2\t{}'.format(quick_dist(conv_pool2.eval(feed_dict=t_dict))))
            print('conv3\t{}'.format(quick_dist(conv_pool3.eval(feed_dict=t_dict))))
            print('fc1\t{}'.format(quick_dist(fc1.eval(feed_dict=t_dict))))
            print('y_conv\t{}\n'.format(quick_dist(y_conv.eval(feed_dict=t_dict))))

            predictions = tf.argmax(y_conv, 1).eval(feed_dict={
                    x: batch[0], keep_prob: 1.0})
            save_distribution(predictions, batch[1], i)
            save_path = saver.save(sess, 'model.ckpt')
        train_step.run(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
          x: test.images(), y_: test.labels(), keep_prob: 1.0}))


def quick_dist(arr):
    """Produces a quick distribution from an array
    """
    return '0%: {:.3f} \t25%: {:.3f} \t50%: {:.3f} \t75%: {:.3f} \t100%: {:.3f}'.format(
        np.percentile(arr, 0),
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
        np.percentile(arr, 100))


def clean():
    print('clearing logdir data, figures, and save files')
    if os.path.exists('logdir/train/'):
        shutil.rmtree('logdir/train/')
    if os.path.exists('logdir/test'):
        shutil.rmtree('logdir/test')
    for f in os.listdir('figs'):
        os.remove(os.path.join('figs/', f))
    if os.path.exists('checkpoint'):
        os.remove('checkpoint')
    if os.path.exists('model.ckpt'):
        os.remove('model.ckpt')
    if os.path.exists('model.ckpt.meta'):
        os.remove('model.ckpt.meta')


if __name__ == '__main__':
    import sys
    args = sys.argv
    if ('--help' in args) or ('-h' in args):
        print(('Available commands:'
        '\n--clean to clear logdir data, figures, and model save files.'
        '\n--wind_speed to train on windspeed and SDO data.'))
        sys.exit()
    if ('--clean' in args):
        clean()
    wind_speed = ('--wind_speed' in args)
    if wind_speed:
        train, test = datareader.get_speed_data()
    else:
        train, test = datareader.get_data_sets()
    main(train, test, wind_speed=wind_speed)
