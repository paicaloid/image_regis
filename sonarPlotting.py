import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

## subplot 4 slot [2x2]
def subplot4(im1, im2, im3, im4, title):
    plt.subplot(221),plt.imshow(im1, cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(im2, cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(im3, cmap = 'gray')
    plt.title(title[2]), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(im4, cmap = 'gray')
    plt.title(title[3]), plt.xticks([]), plt.yticks([])
    plt.show()

def saveplot4(im1, im2, im3, im4, title, filename):
    plt.subplot(221),plt.imshow(im1, cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(im2, cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(im3, cmap = 'gray')
    plt.title(title[2]), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(im4, cmap = 'gray')
    plt.title(title[3]), plt.xticks([]), plt.yticks([])
    plt.savefig(filename)

## subplot 2 slot [1x2] 
def subplot2(im1, im2, title):
    plt.subplot(121),plt.imshow(im1, cmap = 'gray')
    plt.title(title[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(im2, cmap = 'gray')
    plt.title(title[1]), plt.xticks([]), plt.yticks([])
    plt.show()

## plot Cross Power Spectrum
def plotPhase(phaseArray):
    # downscaling has a "smoothing" effect
    lena = scipy.misc.imresize(phaseArray, 0.15, interp='cubic')

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0)

    # show it
    plt.show()