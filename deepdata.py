"""Produces datasets to be fed into inception model
"""


from datetime import datetime, timedelta
import numpy as np
import os
from sunpy.net import hek
from sunpy.time import parse_time
from PIL import ImageOps
import sys
sys.path.append('/sanhome/yshah/Util')
import fetch
sys.path.append('/Users/pauly/repos/aia')
import mov_img
getFitsHdr = mov_img.sun_intensity.getFitsHdr
import multiprocessing as mp


def get_max_flare(result):
    if len(result) == 0:
        return 'N'
    goes_cls = [elem['fl_goescls'] for elem in result]
    cls = max(goes_cls)
    if len(cls) > 0 and cls[0] > 'B':
        return cls[0]
    else:
        return 'N'


def make_aia(start=datetime(2010,5,1), r=[]):
    """Fetch data, then filter based on quality
    """
    if len(r) == 0:
        end = datetime.utcnow()
        delta = timedelta(days=30)
        while end - start > delta:
            r.extend(fetch.fetch('aia.lev1_euv_12s', start=start,
                                 end_or_span=delta, wavelengths=171,
                                 segments='image', cadence=timedelta(hours=6)))
            start += delta
        r.extend(fetch.fetch('aia.lev1_euv_12s', start=start,
                             end_or_span=datetime.utcnow(), wavelengths=171,
                             segments='image', cadence=timedelta(hours=6)))
        print('original query length:', len(r))
        r = [path for path in r if getFitsHdr(path)['quality'] == 0]
        print('filtered query length:', len(r))
    client = hek.HEKClient()
    for i, path in enumerate(r):
        try:
            hdr = getFitsHdr(path)
            tstart = parse_time(hdr['date-obs'])
            tend = tstart + timedelta(days=1)
            result = client.query(hek.attrs.Time(tstart, tend),
                                  hek.attrs.EventType('FL'))
            cls = get_max_flare(result)

            out_path = '/Users/pauly/repos/aia/data/aia/{}/{}'.format(cls,
                    tstart.strftime('%Y-%m-%dT%H:%M:%S'))
            img = mov_img.process_img(path, downscale=(8,8))

            # save image and its symmetries
            img.save('{}_1.jpg'.format(out_path))
            ImageOps.flip(img).save('{}_2.jpg'.format(out_path))
            ImageOps.mirror(img).save('{}_3.jpg'.format(out_path))
            ImageOps.flip(ImageOps.mirror(img)).save('{}_4.jpg'.format(out_path))

            print('\r{}%'.format(int(100*i/len(r))), end='')
        except:
            continue


def make_hmi(start=datetime(2010,5,1), r=[]):
    """Fetch data, then filter based on quality
    """
    if len(r) == 0:
        end = datetime.utcnow()
        delta = timedelta(days=30)
        while end - start > delta:
            r.extend(fetch.fetch('hmi.M_720s', start=start,
                                 end_or_span=delta, segments='magnetogram',
                                 cadence=timedelta(hours=6),
                                 keys=['rsun_obs','cdelt1','quality','date__obs']))
            start += delta
        r.extend(fetch.fetch('hmi.M_720s', start=start,
                             end_or_span=datetime.utcnow(),
                             segments='magnetogram', cadence=timedelta(hours=6),
                             keys=['rsun_obs','cdelt1','quality','date__obs']))
        print('original query length:', len(r))
        r = [sub for sub in r if sub[2] == '0x00000000']
        print('filtered query length:', len(r))
    client = hek.HEKClient()
    for i, sub in enumerate(r):
        try:
            path = sub[-1]
            hdr = getFitsHdr(path)
            tstart = parse_time(sub[3])
            tend = tstart + timedelta(days=1)
            result = client.query(hek.attrs.Time(tstart, tend),
                                  hek.attrs.EventType('FL'))
            cls = get_max_flare(result)

            out_path = '/Users/pauly/repos/aia/data/hmi/{}/{}'.format(cls,
                    tstart.strftime('%Y-%m-%dT%H:%M:%S'))
            img = mov_img.process_hmi(path, float(sub[0]), float(sub[1]), downscale=(8,8))

            # save image and its symmetries
            img.save('{}_1.jpg'.format(out_path))
            ImageOps.flip(img).save('{}_2.jpg'.format(out_path))
            ImageOps.mirror(img).save('{}_3.jpg'.format(out_path))
            ImageOps.flip(ImageOps.mirror(img)).save('{}_4.jpg'.format(out_path))

            print('\r{}%'.format(int(100*i/len(r))), end='')
        except:
            continue


def no_images(r):
    """determines if no good images exist in a query
    """
    out = len(r) == 0
    out = out or r[0] == '' # empty filename check
    out = out or int(r[0][-3], 16) != 0 # quality check
    return out


def make_the_glob(date_range):
    """This code abandoned in favor of using the synoptic data series
    """
    print(''.join(map(chr, [61, 61, 61, 71, 76, 79, 66, 32, 67, 82,
                            69, 65, 84, 73, 79, 78, 32, 73, 78, 73,
                            84, 73, 65, 84, 69, 68, 61, 61, 61])))
    import ipdb
    ipdb.set_trace()
    start, end = date_range
    delta = timedelta(minutes=12)
    last_image_time = start
    image_cache = np.zeros((512,512))
    norm = mov_img.colors.LogNorm(1)
    client = hek.HEKClient()
    while start + delta < end:
        print(start)
        r = fetch.fetch('aia.lev1_uv_24s', wavelengths=1600, start=start,
                        end_or_span=timedelta(minutes=1), segments='image',
                        keys=['quality','date__obs'])
        if not no_images(r):
            print('image found')
            amap = mov_img.Map(r[0][-1])
            amap = mov_img.aiaprep(amap)
            data = mov_img.downscale_local_mean(amap.data, (8,8))
            image_cache = np.sum([image_cache, data], axis=0)
        else:
            start += delta
            continue # this is a spaghetti mess
        if (start - last_image_time > timedelta(hours=6)):
            print('image production triggered.')
            ch3 = np.flipud(norm(image_cache)) # flip images upright and apply log norm
            ch2 = np.flipud(norm(data))
            r = fetch.fetch('hmi.M_720s', start=start,
                            end_or_span=delta, segments='magnetogram',
                            keys=['rsun_obs','cdelt1','quality','date__obs'])
            image_cache = np.zeros((512,512))
            if not no_images(r): # nested if's? what kind of dork programmed this?
                # output image
                ch1 = mov_img.process_hmi(r[0][-1], float(r[0][0]), float(r[0][1]),
                                          downscale=(8,8), single_channel=True)
                result = client.query(hek.attrs.Time(start, start + timedelta(days=1)),
                                      hek.attrs.EventType('FL'))
                cls = get_max_flare(result)
                out_path = '/Users/pauly/repos/aia/data/glob/{}/{}'.format(cls, r[0][-2])
                img = mov_img.misc.toimage(np.array([ch1, ch2, ch3]))

                img.save('{}_1.jpg'.format(out_path))
                ImageOps.flip(img).save('{}_2.jpg'.format(out_path))
                ImageOps.mirror(img).save('{}_3.jpg'.format(out_path))
                ImageOps.flip(ImageOps.mirror(img)).save('{}_4.jpg'.format(out_path))
            print(r[0][-1])
            last_image_time = start
        start += delta


def glob_main():
    date_ranges = [(datetime(x, 5, 1), datetime(x+1, 5, 1)) for x in range(2010,2016)]
    date_ranges.append((datetime(2016,5,1), datetime.utcnow() - timedelta(days=30)))
    with mp.Pool(processes=12) as pool:
        for r in pool.imap_unordered(make_the_glob, date_ranges):
            print('Hodor.')
