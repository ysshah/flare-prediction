import numpy as np
import matplotlib.pyplot as plt
from fetch import fetch
from astropy.io import fits
from mpl_toolkits.mplot3d import Axes3D
import os, glob, shutil, subprocess, datetime, pb0r


S = 4096


def make3Dplot(xyz, uvw=None, vector=True, step=256, movie=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Axis [arcseconds]')
    ax.set_ylabel('Y Axis [arcseconds]')
    ax.set_zlabel('Z Axis [arcseconds]')

    x = xyz[::step,::step,0]
    y = xyz[::step,::step,1]
    z = xyz[::step,::step,2]

    if vector:
        u = uvw[::step,::step,0]
        v = uvw[::step,::step,1]
        w = uvw[::step,::step,2]
        ax.quiver(x, y, z, u, v, w, length=50)
    else:
        ax.scatter3D(x, y, z, c=z)

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1000, 1000)
    ax.set_aspect('equal')

    plt.subplots_adjust(0,0,1,1,0,0)

    if movie:
        for f in glob.glob('Movie/*.png'):
            os.remove(f)
        elevations = np.arange(0, 60, 0.5)
        n = elevations.size
        for i, elev in enumerate(elevations):
            ax.view_init(elev=elev)
            fig.savefig('Movie/frame_{:04d}.png'.format(i), dpi=131, bbox_inches='tight')
            print('\r{}/{} complete.'.format(i+1,n), end='')
        for dst, src in enumerate(range(n-1, -1, -1), n):
            shutil.copyfile('Movie/frame_{:04d}.png'.format(src), 'Movie/frame_{:04d}.png'.format(dst))
        print('\nCopied files in reverse order')
        plt.close(fig)

        cmd = 'ffmpeg -y -r 60 -i {} -c:v libx264 -pix_fmt yuv420p {}'.format(
            os.path.abspath('Movie/frame_%04d.png'), os.path.abspath('Movie/out.mp4'))
        subprocess.call(cmd.split())
        print('\nCreated movie.')
    else:
        print('Saving figure.')
        fig.savefig('plot.png', dpi=131, bbox_inches='tight')
        plt.close(fig)
        # fig.savefig('plot.pdf', bbox_inches='tight')
        # plt.show()
