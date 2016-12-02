"""Script for getting image data and from the datareader.py module and
writing all the images to files. Useful for creating data for inception

"""


import sys
args = sys.argv
if len(args) > 1:
    out_dir = args[1]
else:
    print('ERROR: Please provide the absolute output path')
 

import os
import imp
datareader = imp.load_source('datareader', '../../datareader.py')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


for i in range(4):
    os.mkdir(os.path.join(out_dir, 'label_{}'.format(i)))
images, labels = datareader.get_data_sets(raw=True, channels=3)

"""
data_t = tf.placeholder(tf.uint8)
op = tf.image.encode_jpeg(data_t, format='rgb')
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(len(labels)):
    label_dir = out_dir + 'label_{}/'.format(np.argmax(labels[i]))
    name = label_dir, 'image_{}'.format(i)
    with open('name', 'wb') as f:
        data_np = sess.run(op, images[i].astype(np.uint8))
        f.write(data_np)
"""
for i in range(len(labels)):
    label_dir = out_dir + 'label_{}/'.format(np.argmax(labels[i]))
    name = label_dir + 'image_{}.jpg'.format(i)
    plt.imsave(name, images[i])
