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


label_names = ['no_flare/', 'c_flare/', 'm_flare/', 'x_flare/']

for label in label_names:
    os.mkdir(os.path.join(out_dir, 'train/' + label))
images, labels = datareader.get_data_sets(raw=True, channels=3)

for i in range(len(labels)):
    this_label = label_names[np.argmax(labels[i])]
    label_dir = out_dir + 'train/' + this_label
    name = label_dir + 'image_{}.jpg'.format(i)
    plt.imsave(name, images[i])

def move_files(train_path, label):
    """Moves 20% of the data into a new parent directory with the same structure
    for validation
    """
    lst = os.listdir(train_path + label)
    movers = lst[0:int(len(lst)/5)]
    for fle in movers:
        os.rename(train_path+label+fle, train_path+'../validation/'+label+fle)

for label in label_names:
    move_files(out_dir + 'train/', label)
