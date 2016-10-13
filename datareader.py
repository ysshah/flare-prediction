"""Module for reading curl data into a tensorflow neural net.
Based on the code used for data input for the deep mnist tutorial.
"""

import numpy as np
import pickle
import os
from sunpy.time import parse_time
import bisect


class DataSet(object):
	"""Class for handling data
	Neatly groups x and y data, shuffles data after each epoch, and
	provides functions for retrieving the raw data and metadata
	"""

	def __init__(self, images, labels):
		"""Construct a dataset.
		"""
		assert images.shape[0] == labels.shape[0]
		self._num_examples = images.shape[0]
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def images(self):
		return self._images

	def labels(self):
		return self._labels

	def num_examples(self):
		return self._num_examples

	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""Return the next batch with side 'batch_size' from the data set.
		Shuffles all the data each epoch
		"""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# finished epoch
			self._epochs_completed += 1
			# shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]


def get_data_sets(train_percentage=0.8):
	"""Returns DataSet objects for the train and test data
	train_percentage: decimal indicating the split between the train and test data
	"""
	# get the data in the pickle files and concatenate them together
	print('Getting curls from pickle files...')
	path = '/sanhome/yshah/'
	file_list = [f for f in os.listdir(path) if (f[:6] == 'curls_')]
	curl_list = []
	date_list = []
	for f_str in file_list:
		with open(os.path.join(path, f_str), 'rb') as f:
			data = pickle.load(f)
		curl_list.extend(list(np.nan_to_num(data['curls'])))
		date_list.extend([parse_time(f) for f in data['dates']])
	curls = np.array(curl_list)

	# sort the curls and the dates according to the date order
	print('Sorting curls and dates...')
	order = np.argsort(date_list)
	curls = curls[order]
	date_list = np.array(date_list)[order]

	# TODO: shift the sequence by one and produce more data points with
	# new curls as the most recent one to the flare

	# reshape the data from nx256x256 to n/4 x 256 x 256 x 4
	print('Reshaping data...')
	n_bundles = int(len(curls)/4)
	curls = curls[:n_bundles * 4].reshape(n_bundles, 4, 256, 256)
	curls_reshaped = np.zeros((n_bundles, 256, 256, 4))
	for i in range(len(curls_reshaped)):
		for j in range(4):
			curls_reshaped[i,:,:,j::4] = curls[i,j].reshape(256,256,1)
	date_list_reshaped = date_list[3::4]

	# find train/test divide location and split the curl data
	split = int(train_percentage * len(curls_reshaped))
	train_curls = curls_reshaped[:split]
	test_curls = curls_reshaped[split:]

	# fetch the flare size and create the label data
	flare_a = np.full((len(date_list_reshaped), 4), 0, dtype=np.int)
	flare_a[:,0] = 1
	with open(path + 'hekdatadf.pkl','rb') as f:
		flareData = pickle.load(f)
	flareData = flareData.sort_values('fl_goescls').reset_index()
	for idx in range(len(flareData.index)):
		date = flareData.loc[idx, 'event_peaktime']
		if ((date < date_list_reshaped[-1]) and (date > date_list_reshaped[0])):
			letrCls = flareData.loc[idx, 'fl_goescls']
			sparseCls = let2sparse(letrCls)
			loc = bisect.bisect_left(date_list_reshaped, date)
			if larger_sparse(sparseCls, flare_a[loc]):
				flare_a[loc] = sparseCls
		print('\rFetching flare size data... {}%'.format(
				int(idx * 100 / len(flareData.index))), end='')

	# split the label data
	train_labels = flare_a[:split]
	test_labels = flare_a[split:]

	# feed data into DataSet objects
	print('\nCreating DataSet objects...')
	train = DataSet(train_curls, train_labels)
	test = DataSet(test_curls, test_labels)
	print('Data loading complete.')

	return train, test


def let2sparse(letrCls):
	"""Converts letter designated GOES classes in the format 'M2.4'
	to a sparse array indicating the predicted output like so
	[0,1,0]
	"""
	first = letrCls[0]
	ray = np.zeros(4).astype('int')
	if first == 'C':
		ray[1] = 1
	elif first == 'M':
		ray[2] = 1
	else:
		ray[3] = 1

	return ray


def larger_sparse(f1, f2):
	"""Takes in 2 flare size sparse arrays and returns true
	if the first one is larger
	First one should be a real flare
	"""
	if f2[0] == 1:
		return True
	else:
		indices1, indices2 = np.where(f1==1), np.where(f2==1)
		return (indices1[0][0] > indices2[0][0])


def read_data_sets():
	"""Reads the proprocessed, saved train and test data from a single file.
	"""
	with open('netdata.pkl', 'rb') as f:
		train, test = pickle.load(f)
	return train, test


if __name__ == '__main__':
	"""CURRENTLY NOT WORKING
	Should get the data from get_data_sets and store it in a file,
	for easy access later.
	"""
	train, test = get_data_sets()
	with open('netdata.pkl', 'wb') as f:
		pickle.dump((train, test), f)
	#open(r'netdata.pkl', 'w+b').write((train, test))
