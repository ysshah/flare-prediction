"""Module for reading curl data into a tensorflow neural net.
Based on the code used for data input for the deep mnist tutorial.
"""

import numpy as np
import pickle
import os
from sunpy.time import parse_time
import bisect
from datetime import timedelta
import os
import pandas as pd

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


def get_data_sets(train_percentage=0.8, raw=False, channels=4):
        """Returns DataSet objects for the train and test data for flares
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
        sin_norm = np.zeros((256,256))
        sin_norm[:] = np.sin(np.arange(0,np.pi,np.pi/256))
        sin_norm = np.transpose(sin_norm) # sin norm reduces the values of north/south activity
        curls_reshaped = np.zeros((len(curls)-channels+1, 256, 256, channels))
        for i in range(len(curls) - channels + 1):
                bundle = np.zeros((256, 256, channels))
                bundle[:,:,0] = curls[i] * sin_norm
                date_o = date_list[i]
                for j in range(1, channels):
                        date_t = date_list[i+j]
                        td = date_o - date_t
                        seconds = td.total_seconds() - 21600 * j
                        # append next image if it's within an hour of being 6 hours before
                        if np.abs(seconds) <= 3600:
                                bundle[:,:,j] = curls[i+j] * sin_norm
                        else:
                                dropped += 1
                curls_reshaped[i,:,:,:] = bundle
        dates_reshaped = date_list[:-(channels - 1)]
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

        if raw:
            """Return the raw data"""
            return curls_reshaped, flare_a

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


def get_speed_data(train_percentage=0.8):
        """Data is the 8-channel sdo data generated by Mark
        """
        df = pd.read_csv('omni2_all_years.dat', delim_whitespace=True)
        df.columns = np.arange(1, 56).astype(np.str) # set columns to numbers
        path = '/sanhome/cheung/public_html/AIA/synoptic_ML/'
        pics = [f for f in os.listdir(path) if f[:4] == 'sdo_']
        image_data = np.zeros((len(pics), 256, 256, 8))
        labels = np.zeros((len(pics), 4))
        maxes, maxis = np.zeros(4), np.zeros((4,2)) # for finding the dates for the max values
        cols = ['9', '39', '40', '41']
        for i in range(len(pics)):
                image = np.memmap(os.path.join(path, pics[i]),
                            dtype=np.uint8, mode='r', shape=(128,128,8))
                # upsample to 256x256 for the net
                image_data[i] = image.repeat(2, axis=0).repeat(2, axis=1)
                time = parse_time(pics[i][17:-4]) + timedelta(days=5)
                tt = time.timetuple()
                day = tt.tm_yday
                year = time.year
                # get the label values for the year and day
                dft = df.loc[(df['1']==year) & (df['2']==day) & (df['3']==0)]
                values = np.zeros(4)
                for j in range(4):
                        values[j] = dft[cols[j]].values[0]
                        if values[j] > maxes[j]: ####
                                maxes[j] = values[j]
                                maxis[j] = (dft['1'], dft['2'])
                labels[i] = values
        print(maxis) #######
        # shuffle initial order
        perm = np.arange(len(image_data))
        np.random.shuffle(perm)
        image_data = image_data[perm]
        labels = labels[perm]
        # split data into train and test segments
        split = int(train_percentage * len(image_data))
        # create dataset objects
        train = DataSet(image_data[:split], labels[:split])
        test = DataSet(image_data[split:], labels[split:])
        return train, test


def get_hmi_data():
        # get the list of the files
        # compile list of dates
        # get max flare class for each date
        # move each file to right location based on max flare size
        from scipy.misc import imresize
        from scipy.ndimage import imread
        from scipy.misc import imsave
        from sunpy.net import hek
        client = hek.HEKClient()

        with open('hmi_paths.txt') as f:
                paths = f.readlines()

        for i in range(len(paths)):
                path = paths[i][:-1]
                tstart = parse_time(path[46:56]) + timedelta(hours=int(path[58:60]))
                tend = tstart + timedelta(days=1)
                result = client.query(hek.attrs.Time(tstart, tend),
                                      hek.attrs.EventType('FL'))
                if len(result) == 0:
                        cls = 'N'
                else:
                        goes_cls = [elem['fl_goescls'] for elem in result]
                        max_idx = np.argmax(goes_cls)
                        cls = goes_cls[max_idx]
                        if len(cls) > 0:
                                cls = cls[0]
                        else:
                                cls = 'N'

                img = imread(path)
                img_resize = imresize(img, (256, 256, 3))

                if cls == 'N':
                        folder = 'less_than_c'
                elif cls == 'C':
                        folder = 'c_class'
                elif cls == 'M':
                        folder = 'm_class'
                else:
                        folder = 'x_class'
                imsave('/Users/pauly/hmi_data/{}/{}.jpg'.format(folder, tstart), img_resize)

                print('\r{}%'.format(int(100*i/len(paths))), end='')

























