"""Module for reading curl data into a tensorflow neural net.
Based on the code used for data input for the deep mnist tutorial.
"""

import numpy as np
import pickle
import os
from sunpy.time import parse_time
import bisect


class DataSet(object):

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
	path = '/sanhome/yshah/'
	file_list = [f for f in os.listdir(path) if (f[:6] == 'curls_')]
	curl_list = []
	date_list = []
	for f_str in file_list:
		with open(os.path.join(path, f_str), 'rb') as f:
			data = pickle.load(f)
		curl_list.extend(list(np.nan_to_num(data['curls'])))
		date_list.extend(list(data['dates']))
	curls = np.array(curl_list)

	for idx, date in enumerate(date_list):
		date_list[idx] = parse_time(date)
	
	# sort the curls and the dates according to the date order
	order = np.argsort(date_list)
	curls = curls[order]
	date_list = np.array(date_list)[order]

	##### shift the sequence by one and produce more data points with
	# new curls as the most recent one to the flare

	# reshape the data using np.reshape(len(curls)/4, 256, 256, 4)
	curls_reshaped = curls[:-2].reshape(int(len(curls)/4), 256, 256, 4)
	date_list_reshaped = date_list[3::4]

	###### shuffle?

	split = int(train_percentage * len(curls_reshaped))
	train_curls = curls_reshaped[:split]
	test_curls = curls_reshaped[split:]

	# fetch the flare size and create the label data
	flare_a = np.full((len(date_list_reshaped), 3, 10), 0, dtype=np.int)
	with open('/sanhome/yshah/hekdatadf.pkl','rb') as f:
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
		print('\rLoading data {}%'.format(
				int(idx * 100 / len(flareData.index))), end='')

	train_labels = flare_a[:split]
	test_labels = flare_a[split:]

	train = DataSet(train_curls, train_labels)
	test = DataSet(test_curls, test_labels)

	return train, test


def let2sparse(letrCls):
	"""Converts letter designated GOES classes in the format 'M2.4'
	to a sparse array where each letter, digit is reflected by a 1
	in a certain position of a 3x10 array. eg:
	[0,1,0,0,0,0,0,0,0,0]
	[0,0,1,0,0,0,0,0,0,0]
	[0,0,0,0,1,0,0,0,0,0]
	"""
	first, second, third = letrCls[0], letrCls[1], letrCls[3]
	#ray = np.zeros(22).astype('int')
	ray = np.zeros((3,10)).astype('int')
	if first == 'C':
		#ray[0] = 1
		ray[0,0] = 1
	elif first == 'M':
		#ray[1] = 1
		ray[0,1] = 1
	else:
		#ray[2] = 1
		ray[0,2] = 1
	#ray[int(second) + 2] = 1
	#ray[int(third) + 12] = 1
	ray[1,int(second)] = 1
	ray[2,int(third)] = 1

	return ray


def larger_sparse(f1, f2):
	"""Takes in 2 sparse arrays indicating flare size and returns true
	if the first one is larger
	Frist one should be a real flare
	"""
	indices1, indices2 = np.where(f1==1), np.where(f2==1)
	num1 = int(''.join(str(x) for x in indices1[1]))
	if len(indices2[0]) == 0:
		num2 = 0
	else:
		num2 = int(''.join(str(x) for x in indices2[1]))
	return (num1 > num2)


def read_data_sets():
	with open('netdata.pkl', 'rb') as f:
		train, test = pickle.load(f)
	return train, test


if __name__ == '__main__':
	train, test = get_data_sets()
	with open('netdata.pkl', 'wb') as f:
		pickle.dump((train, test), f)
	#open(r'netdata.pkl', 'w+b').write((train, test))
