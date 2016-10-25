"""Module for reading curl data into a tensorflow neural net.
Based on the code used for data input for the deep mnist tutorial.
"""

import numpy as np
import pickle
import os
from sunpy.time import parse_time
import bisect
from datetime import timedelta


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
	path = '/sanhome/yshah/Curls/'
	file_list = [f for f in os.listdir(path) if (f[:6] == 'curls_')]
	curl_list = []
	date_list = []
	for f_str in file_list:
		with open(os.path.join(path, f_str), 'rb') as f:
			data = pickle.load(f)
		curl_list.extend(list(np.nan_to_num(data['curls'])))
		date_list.extend([parse_time(f) for f in data['dates']])
	curls = np.array(curl_list)

	print('Sorting curls and dates...')
	order = np.argsort(date_list)[::-1]  # descending
	curls = curls[order]
	date_list = np.array(date_list)[order]

	print('Reshaping data... ', end='')
	dropped = 0
	curls_reshaped = np.zeros((len(curls)-3, 256, 256, 4))
	for i in range(len(curls) - 3):
		bundle = np.zeros((256, 256, 4))
		bundle[:,:,0] = curls[i]
		date_o = date_list[i]
		for j in range(1,4):
			date_t = date_list[i+j]
			td = date_o - date_t
			seconds = td.total_seconds() - 21600 * j
			# append next image if it's within an hour of being 6 hours before
			if np.abs(seconds) <= 3600:
				bundle[:,:,j] = curls[i+j]
			else:
				dropped += 1
		curls_reshaped[i,:,:,:] = bundle
	dates_reshaped = date_list[:-3]
	print('{} skipped.'.format(dropped))

	# fetch the flare size and create the label data
	flare_a = np.full((len(dates_reshaped), 4), 0, dtype=np.int)
	flare_a[:,0] = 1
	with open('/sanhome/yshah/hekdatadf.pkl','rb') as f:
		flareData = pickle.load(f)
	flareData = flareData.sort_values('event_peaktime').reset_index()
	for idx in range(len(dates_reshaped)):
		date = dates_reshaped[idx]
		loc_in_flares = bisect.bisect_left(flareData['event_peaktime'], date)
		max_class = 'A'
		while ((loc_in_flares < len(flareData.index)) and
			  (date + timedelta(days=1) >= flareData.loc[loc_in_flares, 'event_peaktime'])):
			class_at_time = flareData.loc[loc_in_flares, 'fl_goescls']
			if (class_at_time > max_class):
				max_class = class_at_time
			loc_in_flares += 1
		flare_a[idx] = let2sparse(max_class)
		print('\rBuilding label data... {}%'.format(
				int(idx * 100 / len(flareData.index))), end='')
	
	# remove data points for which we do not have all of the flare data
	min_fl_date = flareData.loc[0, 'event_peaktime']
	max_fl_date = flareData.loc[len(flareData.index) - 1, 'event_peaktime']
	min_loc = bisect.bisect_left(dates_reshaped, min_fl_date)
	max_loc = bisect.bisect_left(dates_reshaped, max_fl_date - timedelta(days=1))
	curls_reshaped = curls_reshaped[min_loc:max_loc]
	dates_reshaped = dates_reshaped[min_loc:max_loc]
	flare_a = flare_a[min_loc:max_loc]

	# randomize the data order
	print('\nShuffling the data''s initial order')
	perm = np.arange(len(curls_reshaped))
	np.random.shuffle(perm)
	curls_reshaped = curls_reshaped[perm]
	flare_a = flare_a[perm]
	dates_reshaped = dates_reshaped[perm]

	# find train/test divide location and split the data
	print('Splitting the data')
	split = int(train_percentage * len(curls_reshaped))
	train_curls = curls_reshaped[:split]
	test_curls = curls_reshaped[split:]
	train_labels = flare_a[:split]
	test_labels = flare_a[split:]

	# feed data into DataSet objects
	print('Creating DataSet objects...')
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
	if first == 'A':
		ray[0] = 1
	elif first == 'C':
		ray[1] = 1
	elif first == 'M':
		ray[2] = 1
	else:
		ray[3] = 1

	return ray


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
