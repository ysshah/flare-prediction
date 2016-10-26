import numpy as np
import os
import sys
import pickle
from glob import glob
from datetime import datetime, timedelta

import dateutil
import pandas as pd
from skimage.transform import downscale_local_mean
from astropy.io import fits
from sunpy.net import hek

import pb0r
from fetch import fetch

# Image dimension of the HMI data.
IMAGE_DIM = 4096

# Necessary keys and segments to retrieve from JSOC for curl calculations.
KEYS = [
    'CRPIX1',  # Location of sun center in CCD x direction
    'CRPIX2',  # Location of sun center in CCD y direction
    'CDELT1',  # arcsec / pixel in x direction
    'CDELT2',  # arcsec / pixel in y direction
    'DATE__OBS'
]
SEGMENTS = [
    'field',
    'inclination',
    'azimuth',
    'disambig'
]
fetch_args = {
    'dataseries': 'hmi.B_720s',
    'keys': KEYS,
    'segments': SEGMENTS,
    'df': True
}


def printTimeInfo(start, i, N):
    """Print timing statistics of a loop that started at time START, and has
    completed I iterations out of N total iterations.
    """
    stop = datetime.now()
    avg = (stop - start) / i
    remain = avg * (N - i)
    eta = stop + remain
    sys.stdout.write(('\rCompleted {}/{}, avg {:.2f} sec/iter, {:s} remaining,'
        + ' ETA {:%x %I:%M:%S %p}').format(
        i, N, avg.total_seconds(), str(remain).split('.')[0], eta))
    sys.stdout.flush()


def downloadHEKdata(fn='hekdata.pkl'):
    """Download flare data for all flares of magnitude greater than C1, using
    SunPy HEK client. Save output to FN as a pickled Pandas DataFrame.
    """
    FIRST_DATE = datetime(2010, 5, 1)  # First date for which data is available

    # Download 90 days of data at a time
    delta = timedelta(days=90)
    N = (datetime.now() - FIRST_DATE) // delta + 1
    dates = [FIRST_DATE + x*delta for x in range(N)]

    params = (
        hek.attrs.FL,
        hek.attrs.FL.GOESCls >= 'C1',
        hek.attrs.OBS.Instrument.like('goes')
    )

    client = hek.HEKClient()
    flares = pd.DataFrame()

    start = datetime.now()
    for i, date in enumerate(dates, 1):
        r = client.query(hek.attrs.Time(date, date + delta), *params)
        flares = flares.append(pd.DataFrame.from_dict(r), ignore_index=True)
        printTimeInfo(start, i, N)

    with open(fn, 'wb') as f:
        pickle.dump(flares, f)


def getB_cartesian(df, i):
    """Get the B field in cartesian coordinate vectors (x, y, z)."""
    field = fits.open(df.at[i,'field'])[1].data
    tan_inclination = np.tan(np.deg2rad(
        fits.open(df.at[i,'inclination'])[1].data))

    azimuth = np.deg2rad(fits.open(df.at[i,'azimuth'])[1].data)
    disambig = fits.open(df.at[i,'disambig'])[1].data
    flip = (disambig.astype(int) >> 2) % 2 != 0
    azimuth[flip] += np.pi
    tan_azimuth = np.tan(azimuth)

    B_cartesian = np.empty((IMAGE_DIM, IMAGE_DIM, 3))
    B_cartesian[:,:,0] = field * tan_azimuth / np.sqrt(
        tan_inclination**2 + tan_azimuth**2 + 1)
    B_cartesian[:,:,1] = B_cartesian[:,:,0] / tan_azimuth
    B_cartesian[:,:,2] = B_cartesian[:,:,1] * tan_inclination

    return B_cartesian


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

    return curlB


def getCurlOnDate(date):
    """Get the curl of the solar B field on DATE."""
    # Search for 24 minutes in case of missing data
    span = datetime.timedelta(minutes=24)
    df = fetch(start=date, end_or_span=span, **fetch_args)
    return None if df.empty else calculateCurl(df, 0)


def getCurls(folder='/sanhome/yshah/Curls/'):
    """Download curl data from 5/1/2010 to now, and place files in FOLDER.
    Curls are spaced at every 6 hours, and each file contains 50 days of data.
    Will update FOLDER with the latest curls if pre-existing curl files are
    present.
    """
    downscale_factors = (16, 16)
    image_dim_lowres = (IMAGE_DIM // downscale_factors[0],
                        IMAGE_DIM // downscale_factors[1])
    span = timedelta(days=50)
    cadence = timedelta(hours=6)

    if not os.path.isdir(folder):
        os.makedirs(folder)

    currentFiles = sorted(glob(os.path.join(folder, '*.pkl')))
    if currentFiles:
        with open(currentFiles[-1], 'rb') as f:
            data = pickle.load(f)
        first = dateutil.parser.parse(data['dates'][0]).replace(tzinfo=None)
        last = dateutil.parser.parse(data['dates'][-1]).replace(tzinfo=None)
        if (last - first) < (span - cadence):
            N_complete = len(data['dates'])
            print('Updating latest pickle file. '
                'Fetching data from {:%x %X} to {:%x %X}...'.format(
                last + cadence, first + span))
            df = fetch(start=(last + cadence), end_or_span=(first + span),
                cadence=cadence, **fetch_args)
            curls = np.empty((N_complete + df.shape[0], *image_dim_lowres))
            curls[:N_complete] = data['curls']

            start = datetime.now()
            for i in range(df.shape[0]):
                curls[N_complete + i] = downscale_local_mean(
                    calculateCurl(df, i), downscale_factors)
                printTimeInfo(start, i+1, df.shape[0])

            newData = {
                'curls': curls,
                'dates': np.append(data['dates'], np.array(df.DATE__OBS))
            }
            with open(currentFiles[-1], 'wb') as f:
                pickle.dump(newData, f)

            date = first + span
        else:
            date = last + cadence
    else:
        date = datetime(2010, 5, 1)

    while date < datetime.now():
        print('Fetching data from {} to {}...'.format(date, date + span))
        df = fetch(start=date, end_or_span=span, cadence=cadence, **fetch_args)

        curls = np.empty((df.shape[0], *image_dim_lowres))
        start = datetime.now()
        for i in range(df.shape[0]):
            curls[i] = downscale_local_mean(
                calculateCurl(df, i), downscale_factors)
            printTimeInfo(start, i+1, df.shape[0])

        data = {'curls': curls, 'dates': np.array(df.DATE__OBS)}

        with open(os.path.join(folder, 'curls_{:%Y-%m-%d}.pkl'.format(date)), 'wb') as f:
            pickle.dump(data, f)
        print('\nSaved pickle file.')

        date += span
