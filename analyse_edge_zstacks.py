# -*- coding: utf-8 -*-
"""
Analyse a step-function image for resolution.

See: Burns, P. D. Slanted-edge MTF fordigital camera and scanner analysis.
In Is and Ts Pics Conference, 135–138 (SOCIETY FOR IMAGING SCIENCE & TECHNOLOGY, 2000).
http://www.imagescienceassociates.com/mm5/pubs/26pics2000burns.pdf

(c) Richard Bowman 2017, released under GNU GPL

This program expects as input an image containing a single black/white edge.  It will:
1. determine the direction (horizontal/vertical) and sign (black then white or white then black) of the edge
2. measure the angle of the edge (for the analysis to be valid the angle should be close to but not exactly
    horizontal or vertical)
3. average along the edge (or a specified portion thereof) to reduce noise and allow pixel subsampling
4. take the gradient
5. compute the MTF or resolution by a couple of methods

NB a 3-channel colour image is assumed.  Pad grayscale images to be n x m x 3 to avoid wierd results...

"""
from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.patches

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse
import numpy as np
import cv2
from sklearn.cluster import MeanShift

import numpy.fft as fft

import scipy.ndimage
import scipy.interpolate
import itertools

#################### Rotate the image so the bars are X/Y aligned #############
def find_edge_orientation(image, fuzziness = 5):
    """Determine whether the edge in an image is horizontal or vertical
    
    returns: (bool: horizontal, bool: falling)
    The first element of the tuple tells us if the edge is horizontal (true) or vertical (false)
    The second tells us if the edge is low-high (false) or high-low (true).
    """
    edge_scores = []
    gray_image = np.mean(image, axis=2, dtype=np.float)
    for order in [(0,1), (1,0)]: #repeat this twice, taking derivative in X and Y
        edge_image = scipy.ndimage.gaussian_filter(gray_image, order=order, sigma=fuzziness)
        edge_scores.append(np.max(edge_image)) # edge strength assuming it's a rising edge
        edge_scores.append(np.max(-edge_image)) #edge strength assuming it's a falling edge
    orientations = [(h, f) for h in [False, True] for f in [False, True]]
    return orientations[np.argmax(edge_scores)]

def find_edge(image, fuzziness = 5, plot=False):
    """Find the line that best fits an edge

    We reduce images to 2D if they are colour and that the edge is rising and approximately vertical (i.e.
    it goes from low to high as we increase the second array index)"""
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
    ys = np.zeros(image.shape[0])
    for i in range(image.shape[0]):
        ys[i] = np.argmax(scipy.ndimage.gaussian_filter(image[i,:], sigma=fuzziness, order=1))
    xs = np.arange(image.shape[0])
    m = np.polyfit(xs, ys, 1)
    if plot:
        f, ax = plt.subplots(1,1)
        ax.imshow(image.T)
        ax.plot(xs, ys, '.')
        ax.plot(xs, m[0]*xs + m[1])
    return m

def resample_edge(image, line, fuzziness=5, subsampling=5):
    """Resample the edge of an image so it's exactly straight, increasing the resolution to preserve detail.

    image: the (3D) image to resample
    line: [m, c] equation for the y value of the line as a function of x (y=mx + c)
    fuzziness: guess at the PSF width.  We produce a line 10x wider than this (5 sigma either side)
    subsampling: amount to increase resolution to allow sub-pixel sampling. 4-10 is about right, I think.

    returns: an Nx(10 x fuzziness x subsampling) x 3 array with the line exactly halfway along it.
    """
    edge = np.zeros((image.shape[0], fuzziness*10*subsampling, image.shape[2]))
    for i in range(image.shape[0]):
        y = line[0]*i + line[1]
        for j in range(subsampling):
            ystart = int(y + j/subsampling) - 5*fuzziness
            edge[i, j::subsampling, :] = image[i,ystart:ystart + fuzziness*10, :]
    return edge

def find_psf(image, fuzziness=5, subsampling=1, show_plots=False):
    """Given an oriented image, calculate the PSF from the edge response.

    Returns a (10*fuzziness-1, 3) array with the PSF in 3 colours.
    """
    m = find_edge(image, plot=show_plots)  # locate the edge (this is the equation of a straight line
    if subsampling > 1000:
        assert np.abs(m[0]*image.shape[0]) < 1, "Error: to use subsampling you must have a slanted " \
                                                "edge, {} is too straight for image {}!".format(m, image.shape)
        if np.abs(m[0]*image.shape[0]) < 10: print("Warning: line {} is insufficiently tilted"
                                           "for image {}, PSF may have pixel bias.".format(m, image.shape))
    edge = resample_edge(image, m, fuzziness=fuzziness, subsampling=subsampling)
    psf_image = np.diff(edge, axis=1)
    psf = np.mean(psf_image, axis=0)  # this should average over the line and get the PSFs
    return psf, m

def plot_psf(psf, ax=None, x=None, xlabel="position/pixels", subsampling=1):
    if ax is None:
        f, ax = plt.subplots(1,1)
    if x is None:
        x = np.arange(psf.shape[0])/subsampling
    for i, col in enumerate(['red', 'green', 'blue']):
        ax.plot(x, psf[:, i], color=col)
    ax.set_xlabel(xlabel)

def inset_image(fig, image, line=None, horizontal=False, flip_line=False):
    ax2 = fig.add_axes((0.75, 0.70, 0.15, 0.15), frameon=False)
    ax2.imshow(image)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    if line is not None:
        x = np.array([0, image.shape[1 if horizontal else 0]])
        y = x*line[0] + line[1]
        if flip_line:
            y = image.shape[0 if horizontal else 1] - y + 1
        if not horizontal:
            x, y = y, x
        ax2.plot(x, y, 'r-')

def find_fwhm(psf, annotate_ax=None, interp=10, subsampling=1):
    ss = subsampling
    fwhm = np.zeros(psf.shape[1])
    for i in range(fwhm.shape[0]):
        y = psf[:,i]
        if interp>1: #use basic linear interpolation to improve resolution...
            x = np.arange(len(y))
            xi = np.arange(len(y)*interp)/interp
            y = np.interp(xi, x, y)
        threshold = np.max(y) / 2
        ileft = np.argmax(y > threshold)/interp
        iright = (len(y) - 1 - np.argmax(y[::-1] > threshold))/interp
        fwhm[i] = iright - ileft
        if annotate_ax is not None:
            h = np.max(psf)
            annotate_ax.annotate("fwhm {0:.1f}".format(fwhm[i]/ss),
                                 xy=(ileft/ss, y[int(iright*interp)]),
                                 xytext=(3/ss, h/2 + (i-1)*h/10),
                                 arrowprops = dict(facecolor=annotate_ax.lines[i].get_color(), shrink=0))
    return fwhm/ss

if __name__ == '__main__':
    import os.path
    import os
    from skimage.io import imread
    from matplotlib.backends.backend_pdf import PdfPages
    dir = os.path.dirname(__file__)
    fnames = [n for n in os.listdir(dir) if n.startswith("edge") and n.endswith(".jpg")]
    with PdfPages(os.path.join(dir, "edge_function_analysis.pdf")) as pdf:
        for fname in fnames:
            print("Processing {}...".format(fname),end="")
            image = imread(os.path.join(dir, fname))
            original_image = image
            horizontal, falling = find_edge_orientation(image)  # figure out the direction of the edge
            if horizontal:
                image = image.transpose((1,0,2))  # if the edge is in the first array index, move it to the second
            if falling:
                image = image[:, ::-1, ...]  # for falling edges, flip the image so they're rising

            h, w, d = image.shape
            image = image[2*h//5:3*h//5, :, ...]  # crop out the middle 20% of the image for now

            ss = 5

            psf, line = find_psf(image, fuzziness=5, subsampling=ss)
            mtf = np.abs(fft.rfft(psf, axis=0, n=psf.shape[0]))

            f, ax = plt.subplots(1,1)
            plot_psf(psf, ax=ax, subsampling=ss)
            find_fwhm(psf, ax, subsampling=ss)
            inset_image(f, original_image, line, horizontal, falling)
            ax.set_title("PSF for "+fname)
            #f.savefig(os.path.join(dir, "psf_"+fname))
            pdf.savefig(f)
            f, ax = plt.subplots(1,1)
            plot_psf(mtf, ax=ax, x=fft.rfftfreq(psf.shape[0])*ss, xlabel="spatial frequency * 1 pixel")
            inset_image(f, original_image, line, horizontal, falling)
            ax.set_title("MTF for "+fname)
            #f.savefig(os.path.join(dir, "mtf_"+fname))
            pdf.savefig(f)
            print("done")

"""    if show_plots:
        f, ax = plt.subplots(1,2)
        for i, col in enumerate(['red', 'green', 'blue']):
            ax[0].plot(psf[:,i], color=col)
            ax[1].plot(fft.rfftfreq(psf.shape[0]), mtf[:,i], color=col)
        ax[0].set_xlabel('position/pixels')
        ax[1].set_xlabel('spatial frequency/pixels^-1')
    plt.show()
"""
