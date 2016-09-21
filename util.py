from datetime import datetime, timedelta
from sunpy.net import hek
import pandas as pd
import pickle
import sys


def getTimeInfo(start, i, N):
    """Get timing statistics of a loop that started at time START, and has
    completed I iterations out of N total iterations."""
    stop = datetime.now()
    avg = (stop - start) / i
    remain = avg * (N - i)
    eta = stop + remain
    sys.stdout.write(('\rCompleted {}/{}, avg {:.2f} sec/iter, {:s} remaining, '
        + 'ETA {:%x %I:%M:%S %p}').format(
        i, N, avg.total_seconds(), str(remain).split('.')[0], eta))
    sys.stdout.flush()


def downloadHEKdata(fn='hekdata.pkl'):
    """Download flare data using SunPy HEK client, save output to FN."""
    FIRST_DATE = datetime(2010, 5, 1) # First date for which data available
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
        getTimeInfo(start, i, N)

    with open(fn, 'wb') as f:
        pickle.dump(flares, f)
