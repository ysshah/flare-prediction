import numpy as np
# import matplotlib.pyplot as plt
from util import getTimeInfo
from astropy.io import fits
from skimage.transform import downscale_local_mean
from fetch import fetch
from datetime import datetime, timedelta
from scipy import misc
# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont
# import tensorflow as tf
import pandas as pd
import pickle
import bisect
import pb0r
# import matplotlib
# import sunpy.cm

# Image dimension of the HMI data.
IMAGE_DIM = 4096
N_CLASSES = 4
KEYS = [
    'CRPIX1', # Location of sun center in CCD x direction
    'CRPIX2', # Location of sun center in CCD y direction
    'CDELT1', # arcsec / pixel in x direction
    'CDELT2', # arcsec / pixel in y direction
    'DATE__OBS'
]
SEGMENTS = [
    'field',
    'inclination',
    'azimuth',
    'disambig'
]


def getFlareData(fn='/sanhome/yshah/hekdatadf.pkl'):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    print('Loaded flare data.')
    return data


def loadJSOCdata():
    with open('/sanhome/yshah/JSOC_data/data.pkl', 'rb') as f:
        df = pickle.load(f)
    return df


def findClosestTimes(df, time, dates=None):
    if not dates:
        dates = list(df['DATE__OBS'])
    prev = datetime.timedelta(days=1)
    idx = np.array([bisect.bisect_left(dates, time[i] - prev
        ) for i in range(len(time))])
    return idx


def addFlareColumnToJSOC():
    jsoc = loadJSOCdata()
    print('Loaded JSOC data.')
    flares = getFlareData()

    jsoc['flare'] = pd.Series()
    dates = list(jsoc['DATE__OBS'])
    back = datetime.timedelta(days=1)
    for i in range(flares.shape[0]):
        t = flares.loc[i, 'event_peaktime']
        start = bisect.bisect_left(dates, t - back)
        end = bisect.bisect_left(dates, t)
        jsoc.loc[start:end, 'flare'] = flares.loc[i, 'fl_goescls']
        print('\r{}/{}'.format(i, flares.shape[0]), end='')


def genCurlPlots():
    date = datetime.datetime(2012, 3, 6, 8, 36, 12)
    end = datetime.datetime(2012, 3, 9)

    df = fetch(dataseries='hmi.B_720s', start=date, end_or_span=span,
        keys=KEYS, df=True, segments=SEGMENTS, parse_dates=['DATE__OBS'])
    print('Fetched data, {} images total.'.format(df_aia.shape[0]))

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 12)
    start = datetime.datetime.now()
    for i in range(df.shape[0]):
        curl = downscale_local_mean(calculateCurl(df, i), (16,16))

        norm = matplotlib.colors.SymLogNorm(1, vmin=np.nanmin(curl), vmax=np.nanmax(curl))

        pil_img = misc.toimage(matplotlib.cm.viridis(norm(np.fliplr(curl))))
        draw = ImageDraw.Draw(pil_img)
        draw.text((12, 12), 'Curl {:%Y-%m-%d %H:%M:%S}'.format(df.at[i, 'DATE__OBS']), font=font)
        pil_img.save('Movie2/frame_{:04d}.png'.format(i))
        getTimeInfo(start, i+1, df.shape[0])


def genAIAplots():
    date = datetime.datetime(2012, 3, 6, 8, 36, 12)
    end = datetime.datetime(2012, 3, 9)

    df = fetch('aia.lev1_euv_12s', start=date, end_or_span=end, wavelengths=171,
        cadence=datetime.timedelta(minutes=12), keys=['DATE__OBS','WAVELNTH'],
        segments='image', df=True, parse_dates=['DATE__OBS'])
    print('Fetched data, {} images total.'.format(df.shape[0]))

    cmap = sunpy.cm.get_cmap('sdoaia171')
    cmap.set_bad()  # Set masked values to black color
    norm = matplotlib.colors.LogNorm(vmin=10, vmax=6000, clip=True)

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 12)
    start = datetime.datetime.now()
    for i in range(df.shape[0]):
        f = fits.open(df.at[i,'image'])
        f[1].verify('silentfix')
        data = np.flipud(downscale_local_mean(f[1].data, (16,16)))
        f.close()
        pil_img = misc.toimage(cmap(norm(data)))
        draw = ImageDraw.Draw(pil_img)
        draw.text((12, 12), 'AIA 171 {:%Y-%m-%d %H:%M:%S}'.format(df.at[i,'DATE__OBS']), font=font)
        pil_img.save('Movie3/frame_{:04d}.png'.format(i+158))
        getTimeInfo(start, i+1, df.shape[0])


def makeCurlMovie():
    cmd = 'ffmpeg -y -r 30 -i {} -c:v libx264 -pix_fmt yuv420p {}'.format(
        os.path.abspath('Movie2/frame_%04d.png'), os.path.abspath('Movie2/out.mp4'))
    subprocess.call(cmd.split())


def combine():
    for i in range(3):
        curl = Image.open('Movie2/frame_{:04d}.png'.format(i))
        aia = Image.open('Movie3/frame_{:04d}.png'.format(i))

        new_image = Image.new('RGB', (512, 256))
        new_image.paste(curl, (0, 0))
        new_image.paste(aia, (256, 0))
        new_image.save('Movie4/frame_{:04d}.png'.format(i))

        curl.close()
        aia.close()


def getB_cartesian(df, i):
    """Get the B field in cartesian coordinate vectors (x, y, z)."""
    field = fits.open(df.at[i,'field'])[1].data
    tan_inclination = np.tan(np.deg2rad(fits.open(df.at[i,'inclination'])[1].data))

    azimuth = np.deg2rad(fits.open(df.at[i,'azimuth'])[1].data)
    disambig = fits.open(df.at[i,'disambig'])[1].data
    flip = (disambig.astype(int) >> 2) % 2 != 0
    azimuth[flip] += np.pi
    tan_azimuth = np.tan(azimuth)

    B_cartesian = np.empty((IMAGE_DIM, IMAGE_DIM, 3))
    B_cartesian[:,:,0] = field * tan_azimuth / np.sqrt(tan_inclination**2 + tan_azimuth**2 + 1)
    B_cartesian[:,:,1] = B_cartesian[:,:,0] / tan_azimuth
    B_cartesian[:,:,2] = B_cartesian[:,:,1] * tan_inclination

    return B_cartesian


def getCurlOnDate(date):
    """Get the curl of the solar B field on DATE."""

    # Search for 24 minutes in case of missing data
    span = datetime.timedelta(minutes=24)
    df = fetch(dataseries='hmi.B_720s', start=date, end_or_span=span,
        keys=KEYS, df=True, segments=['field', 'inclination', 'azimuth',
        'disambig'])

    if df.empty:
        return None

    return calculateCurl(df, 0)


def calculateCurl(df, i):
    """Calculate the curl of the B field."""
    date = df.at[i,'DATE__OBS']
    cx = int(df.at[i,'CRPIX1'])
    dx = df.at[i,'CDELT1']
    cy = int(df.at[i,'CRPIX2'])
    dy = df.at[i,'CDELT2']

    # Get solar radius on DATE and convert arcseconds to pixels
    radius = pb0r.pb0r(date, arcsec=True)['sd'].value
    radius_pix_x = np.ceil(radius / dx).astype(int)
    radius_pix_y = np.ceil(radius / dy).astype(int)

    # Create phi and theta arrays of size 4096 x 1
    phi, theta = np.empty(IMAGE_DIM), np.empty(IMAGE_DIM)
    phi[cx-radius_pix_x:cx+radius_pix_x] = np.linspace(
        -np.pi/2, np.pi/2, radius_pix_x*2)
    theta[cy-radius_pix_y:cy+radius_pix_y] = np.linspace(
        0, np.pi, radius_pix_y*2)

    # Create R vector = (R_x, R_y, R_z)
    R = np.empty((IMAGE_DIM, IMAGE_DIM, 3))
    R[:,:,0] = np.append(np.arange(-cx,0), np.arange(IMAGE_DIM-cx)) * dx
    R[:,:,1] = np.append(np.arange(-cy,0),
                         np.arange(IMAGE_DIM-cy)).reshape(-1,1)[::-1] * dy
    R[:,:,2] = np.sqrt(radius**2 - R[:,:,0]**2 - R[:,:,1]**2)

    # Create phi vector = (cos(phi), 0, -sin(phi))
    phi_vec = np.empty((IMAGE_DIM, IMAGE_DIM, 3))
    phi_vec[:,:,0] = np.cos(phi)
    phi_vec[:,:,1] = 0
    phi_vec[:,:,2] = -np.sin(phi)

    # Create theta vector = phi vector x R vector
    theta_vec = np.cross(phi_vec, R)

    # Get phi and theta components of B field
    B_cartesian = getB_cartesian(df, i)
    B_phi = np.einsum('ijk,ijk->ij', phi_vec, B_cartesian)
    B_theta = np.einsum('ijk,ijk->ij', theta_vec, B_cartesian)

    # Calculate derivatives of B_phi and B_theta with respect to theta, phi
    dphi = np.tile(np.gradient(phi), (theta.size, 1))
    dtheta = np.tile(np.gradient(theta).reshape(-1,1), (1, phi.size))
    dBphi_dtheta, dBphi_dphi = np.gradient(B_phi, dtheta, dphi)
    dBtheta_dtheta, dBtheta_dphi = np.gradient(B_theta, dtheta, dphi)

    # Calculate the curl
    theta_col = theta.reshape(-1,1)
    curlB = 1 / (radius * np.sin(theta_col)) * (
        np.cos(theta_col * B_phi) * (B_phi + theta_col*dBphi_dtheta)
        - dBtheta_dphi)

    # Old code... wrong answer
    # B_cartesian[:,:,0] = field / np.sqrt(3) * np.sin(azimuth_rad) * np.cos(inclination_rad)
    # B_cartesian[:,:,1] = field / np.sqrt(3) * np.sin(azimuth_rad) * np.sin(inclination_rad)
    # B_cartesian[:,:,2] = field / np.sqrt(3) * np.cos(azimuth_rad)

    return curlB


def getAllCurls():
    """Get all curls of B fields."""
    ONEDAY = datetime.timedelta(days=1)

    days_span = 2242 # Number of days over which to download data.

    flares = getFlareData()

    flareDates = pd.DatetimeIndex(flares.event_peaktime).normalize().astype('O')
    allDates = np.array([datetime.datetime(2010, 5, 1)
                         + datetime.timedelta(days=x) for x in range(days_span)])
    noFlareDates = np.setdiff1d(allDates, flareDates)

    curls = []
    classes = []

    print('Total flares: {}; total no dates: {}'.format(
        flares.shape[0], noFlareDates.size))

    start = datetime.datetime.now()
    for i, row in enumerate(flares[['event_peaktime', 'fl_goescls']].itertuples()):
        curl = getCurlOnDate(row.event_peaktime.to_pydatetime() - ONEDAY)
        if curl is None:
            print('\nWarning: No previous data for {} class flare on {}'.format(
                row.fl_goescls, row.event_peaktime))
        else:
            curls.append(curl)
            classes.append(row.fl_goescls[0])
        print('\rFlares: ' + getTimeInfo(start, i+1, flares.shape[0]), end='')

    print()

    start = datetime.datetime.now()
    for i, date in enumerate(noFlareDates):
        curl = getCurlOnDate(date - ONEDAY)
        if curl is None:
            print('\nWarning: No previous data for date {}'.format(date - ONEDAY))
        else:
            curls.append(curl)
            classes.append('')
        print('\rNo flares: ' + getTimeInfo(start, i+1, noFlareDates.size), end='')

    classes = np.array(classes)
    sparseClasses = np.zeros((classes.size, N_CLASSES))

    sparseClasses[classes == '' , 0] = 1
    sparseClasses[classes == 'C', 1] = 1
    sparseClasses[classes == 'M', 2] = 1
    sparseClasses[classes == 'X', 3] = 1

    data = {'curls': np.array(curls), 'classes': sparseClasses}

    with open('dnn_data', 'wb') as f:
        pickle.dump(data, f)
    print('\nSaved pickle file.')


def getCurls():
    date = datetime(2010, 5, 1)
    span = timedelta(days=50)
    cadence = '6h'

    log = open('/sanhome/yshah/log.txt', 'w')

    while date < datetime.now():
        df = fetch(dataseries='hmi.B_720s', start=date, end_or_span=span,
            keys=KEYS, cadence=cadence, df=True, segments=SEGMENTS)

        curls = np.empty((df.shape[0], 256, 256))
        start = datetime.now()
        for i in range(df.shape[0]):
            try:
                curls[i] = downscale_local_mean(calculateCurl(df, i), (16,16))
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                log.write('Error on date {}\n'.format(df.at[i,'DATE__OBS']))
                curls[i] = np.ones((256,256)) * -1
            getTimeInfo(start, i+1, df.shape[0])

        data = {'curls': curls, 'dates': np.array(df.DATE__OBS)}

        with open('/sanhome/yshah/curls_{:%Y-%m-%d}.pkl'.format(date), 'wb') as f:
            pickle.dump(data, f)
        print('\nSaved pickle file.')

        date += span

    log.close()


if __name__ == '__main__':

    getCurls()

    # with open('dnn_data', 'rb') as f:
    #     data = pickle.load(f)

    # curls = [curl.flatten() for curl in data['curls']]
    # classes = data['classes']

    # x = tf.placeholder(tf.float32, [None, IMAGE_DIM**2])

    # W = tf.Variable(tf.zeros([IMAGE_DIM**2, N_CLASSES]))
    # b = tf.Variable(tf.zeros([N_CLASSES]))

    # y = tf.nn.softmax(tf.matmul(x, W) + b)

    # y_ = tf.placeholder(tf.float32, [None, N_CLASSES])

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
    #                                               reduction_indices=[1]))

    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)

    # batch_xs = curls
    # batch_ys = classes

    # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # for i in range(1000):
    #     batch_xs = np.random.rand(100, 784)
    #     batch_ys = np.random.rand(100, 10)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #     print('\r{}/1000'.format(i+1), end='')

    # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
