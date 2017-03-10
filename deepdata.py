"""Produces datasets to be fed into inception model
"""


from datetime import datetime, timedelta
import numpy as np
import os
from sunpy.net import hek
from sunpy.time import parse_time
from PIL import ImageOps
import imp
fetch = imp.load_source('fetch', '/sanhome/yshah/Util/fetch.py')
try:
    import mov_img
except ImportError:
    print('mov_img direct import unavailable. Importing from ../aia/')
    mov_img = imp.load_source('mov_img', '../aia/mov_img.py')
getFitsHdr = mov_img.sun_intensity.getFitsHdr


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
