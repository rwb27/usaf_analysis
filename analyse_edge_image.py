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
3. align rows in the image so that the edge is always at x=0
4. combine all the rows together using a smoothing spline
4. take the gradient
5. compute the MTF or resolution by a couple of methods

NB a 3-channel colour image is assumed.  Pad grayscale images to be n x m x 3 to avoid wierd results...

"""
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches
from matplotlib.gridspec import GridSpec

# from skimage import data
# from skimage.feature import corner_harris, corner_subpix, corner_peaks
# from skimage.transform import warp, AffineTransform
# from skimage.draw import ellipse
import numpy as np
import cv2
# from sklearn.cluster import MeanShift

import numpy.fft as fft

import scipy.ndimage
import scipy.interpolate
import scipy.optimize
from scipy.interpolate import LSQUnivariateSpline
import itertools
import os
import sys
import re
import os.path
import yaml
from skimage.io import imread
from matplotlib.backends.backend_pdf import PdfPages
from analyse_distortion import find_edge_orientation, find_edge, reduce_1d
from extract_raw_image import load_raw_image
import analyse_distortion

#################### Rotate the image so the bars are X/Y aligned #############

def reorient_image(image, vertical, falling):
    """Flip or transpose an image.
    
    image: a 3D array representing a (colour) image
    horizontal: whether the image should be transposed
    falling: whether the image should be mirrored
    
    Returns: image, flipped/transposed as requested.
    """
    if vertical:
        image = image.transpose((1,0,2))  # if the edge is in the first array index, move it to the second
    if falling:
        image = image[:, ::-1, ...]  # for falling edges, flip the image so they're rising
    return image
    
def locate_edge(image, fuzz=10, threshold=0.3):
    """Fit the peak gradients of the image with a straight line.
    
    A non-directional first derivative of the image is taken (by applying
    first-order Gaussian filters along X and Y, then summing the squares
    of the results).  This should yield an image where the edge is a 
    bright line.  A straight line is fitted to these pixels, using the
    first and second moments of the intensity distribution.
    
    Later analysis functions expect the edge to be roughly horizontal 
    (i.e. image[i,:] should be a step function) and increasing (so the
    step function goes from black to white).  If the image should be 
    transposed or flipped, "vertical" or "falling" will be set to True,
    and the line returned will refer to the flipped/transposed image.
    
    Arguments:
        image: numpy.array
            a 2- or 3-dimensional numpy array representing the image.  Colour
            images are converted to grayscale.
        fuzz: float (optional, default 10)
            the smoothing parameter in pixels, used for edge detection.
        threshold: float (optional, default 0.3)
            a cut-off value applied to distinguish the edge from
            background noise, as a fraction of the maximum gradient value.
    
    Returns: (horizontal, line)
        vertical: bool
            Whether the line is closer to vertical than horizontal.  If so, later
            analysis will require the image to be transposed so the line is horizontal.
        falling: bool
            If the edge runs white-to-black as y is increased, we will need to flip
            the image along the y axis to ensure later functions are consistent.
        line: tuple of two floats (gradient, intercept)
            The equation of the line.  y = x*gradient + intercept (the same as if we
            had used ``numpy.polyfit()`` with ``order=1``).  NB x and y here refer
            to the first and second indices of the image **after** flipping/transposing
            as indicated by the first two variables
    """
    gray_image = image.mean(axis=2) if len(image.shape) == 3 else image
    # start with a non-directional edge detection
    from scipy.ndimage.filters import gaussian_filter
    edges = gaussian_filter(gray_image, fuzz, order=(0,1))**2
    edges += gaussian_filter(gray_image, fuzz, order=(1,0))**2
    # background-subtract to remove bias due to nonzero background
    edges -= edges.max()*threshold
    edges[edges<0] = 0
    # calculate moments of the image
    x = np.arange(edges.shape[0])[:,np.newaxis]
    y = np.arange(edges.shape[1])[np.newaxis,:]
    def expectation(function, weights):
        return np.mean(function * weights)/np.mean(weights)
    cx = expectation(x, edges)
    cy = expectation(y, edges)
    Sxx = expectation((x-cx)**2, edges)
    Syy = expectation((y-cy)**2, edges)
    Sxy = expectation((x-cx)*(y-cy), edges)
    # For later analysis, it makes sense to ensure the edge is consistently oriented.
    # First, make sure the edge is ~horizontal (the image should be transposed if not)
    vertical = Syy > Sxx # True if the line is closer to x=const than y=const
    if vertical: # the image should be transposed if the line isn't horizontal
        x, y, cx, cy, Sxx, Syy = x.T, y.T, cy, cx, Syy, Sxx
        gray_image = gray_image.T
    gradient = Sxy/Sxx
    intercept = cy - cx * gradient
    # Next, make sure the edge is rising (black-to-white), image should be flipped in Y if not.
    falling = np.sum(gray_image[y < x*gradient + intercept]) > 0.5*np.sum(gray_image)
    if falling: # Update the line so it is correct for the flipped image
        intercept = np.max(y) - intercept
        gradient = -gradient
    return vertical, falling, (gradient, intercept)

def deduce_bayer_pattern(image, unit_cell=2):
    """Deduce which pixels in an image are nonzero, modulo a unit cell.
    
    image: a 3D numpy array corresponding to a colour image.
    unit_cell: either a 2-element tuple or a scalar (for square unit cells).
    
    Returns: 3D numpy array of booleans, showing which elements of the unit
        cell are nonzero for each colour channel.
    
    The image should have one plane per colour channel (any number of channels
    is ok), and only pixels with the appropriate colour filter should be
    nonzero.  I.e. you'd expect for a standard RGGB pattern that 3/4 of the
    red and blue pixels are zero, and 2/4 of the green pixels are zero.
    * If bayer_pattern is a Boolean array, it should have the dimensions
    of the unit cell, and the number of colour planes of the image.  E.g.
    for a standard RGGB pattern, it should be 2x2x3.
    """
    if np.isscalar(unit_cell):
        unit_cell = (unit_cell, unit_cell)
    bayer_pattern = np.zeros(tuple(unit_cell) + (image.shape[2],), dtype=np.bool)
    w, h = unit_cell
    for i in range(w):
        for j in range(h):
            bayer_pattern[i,j,:] = np.sum(np.sum(image[i::w, j::h, :], axis=0), axis=0) > 0
    return bayer_pattern
    

def extract_aligned_edge(raw_image, width, channel=1, force_tx=None, thresholds=(0.25,0.75)):
    """Extract x, I coordinates for each row of the image, aligned to the edge
    
    raw_image should be an NxMx3 RGB image, with the dark-bright transition along the middle axis
    width will be the width of the output rows (should be less than the width of the raw image)
        NB this width is relative to the original image - as each row will only contain the 
        pixels with the selected colour, the rows will have half this many points.
    channel is the channel of the image to extract (0-2)
    thresholds specifies the region of I (relative to min/max)
    
    Returns: [(x, I, tx)]
    x and I are cropped 1D arrays, each representing the x and I coordinates for one row.  The range of x
    will be approximately -widdth/2 to width/2, though there will usually be width/2 points.  tx is the 
    x coordinate where the transition was detected (i.e. the shift between the x coordinates and the pixels
    of the original image).
    """
    bayer = deduce_bayer_pattern(raw_image)[:,:,channel]
    aligned_rows = []
    for i in range(raw_image.shape[0]):
        row_bayer = bayer[i % bayer.shape[0], :]
        if np.any(row_bayer): # if there are active pixels of this channel in this row
            x = np.argmax(row_bayer) # find the position of the first relevant pixel in the row
            active_pixel_slice = slice(x, None, bayer.shape[1]) 
                # This slice object takes every other pixel, starting at x (x==0 or 1)

            # Extract the position and intensity of all of the pixels that are active
            x = np.arange(raw_image.shape[1])[active_pixel_slice]
            I = raw_image[i,active_pixel_slice,channel]

            # Crop out the part where it goes from black to white (from 25% to 75% intensity)
            normI = (I - np.min(I) + 0.0)/(np.max(I)-np.min(I)) # NB the +0.0 converts to floating point
            start = np.argmax(normI > thresholds[0])
            stop = np.argmax(normI > thresholds[1])

            # Fit a line and find the point where it crosses 0.5
            gradient, intercept = np.polyfit(x[start:stop], normI[start:stop], 1)
            transition_x = (0.5 - intercept)/gradient # 0.5 = intercept + gradient*xt
            
            # Now, crop out a region centered on the transition
            start = np.argmax(x > transition_x - width/2.0)
            #stop = np.argmax(x > transition_x + width/2.0)
            stop = start + width//row_bayer.shape[0] # Should do the same as above, but guarantees length.
            aligned_rows.append((x[start:stop] - transition_x, I[start:stop], transition_x))
    return aligned_rows

def sorted_x_and_I(aligned_rows):
    """Extract all x and I coordinates from a set of rows, in ascending x order
    
    Arguments: 
        aligned_rows: the output from extract_aligned_edge
        
    Returns:
        sorted_x, sorted_I, txs

    1D numpy arrays of x and I, in ascending x order, and an array of all the original x positions
    (the mean of txs tells you where the edge was before alignment)
    """
    # First, extract the x and I coordinates into separate arrays and flatten them.
    xs, Is, txs = zip(*aligned_rows)
    all_x = np.array(xs).flatten()
    all_I = np.array(Is).flatten()

    # Points must be sorted in ascending order in x
    order = np.argsort(all_x)
    sorted_x = all_x[order]
    sorted_I = all_I[order]

    # If any points are the same, spline fitting fails - so add a little noise
    while np.any(np.diff(sorted_x) <= 0):
        i = np.argmin(np.diff(sorted_x))
        sorted_x[i+1] += 0.0001 # 0.0001 is in units of pixels, i.e. insignificant.
    return sorted_x, sorted_I, txs

def average_edge_spline(sorted_x, sorted_I, dx=0.1, knot_spacing=2.0, crop=10):
    """Average the edge, using a spline to smooth it.

    Arguments:
        sorted_x: a 1D array, in ascending order, of x coordinates
        sorted_I: a corresponding array of intensity coordinates
        dx: the resolution at which to interpolate (default: 0.1 pixel)
        knot_spacing: distance between the knots of the interpolating spline.
            (default: 2.0).  This sets the smoothness we require, larger=smoother.
        crop: the returned interpolated array will be smaller than the input arrays
            by this amount (default 10 pixels).  NB this is in the same units as
            sorted_x, and doesn't correspond either to points in sorted_x or points
            in the returned interpolated array.
    
    Returns: interpolated_x, interpolated_I
        An array of interpolated x coordinates and an array of correspoinding I values.
    """
    xmin, xmax = np.min(sorted_x), np.max(sorted_x)
    ks = knot_spacing
    spline = LSQUnivariateSpline(sorted_x, sorted_I, np.arange(xmin + ks, xmax - ks, ks))
    sx = np.arange(xmin + crop, xmax - crop, dx)
    return sx, spline(sx)

def numerical_diff(x, y, crop=0, sqrt=False):
    """Numerically differentiate a vector of equally spaced values

    Arguments:
        x: the independent variable, assumed to be evenly spaced
        y: the dependent variable to be differentiated
        crop: return a shorter array than the input by removing
            this many points from the start and end (default: 0)
        sqrt: instead of calculating d/dx y, calculate (d/dx y**0.5)**2
            to recover the point spread function from an edge (because
            it's the underlying complex function, and not the intensity,
            that we really want to differentiate).
    
    Returns: x_midpoints, diff_y
        The two 1D arrays are the mid-points of the x coordinates (i.e. 
        the X coordinates of the differentiated y values), and the 
        numerically differentiated y values (scaled by the difference 
        between x points to approximate the true derivative).
    """
    crop = np.argmax(x > np.min(x) + crop)
    if crop > 0:
        cropped_x = x[crop:-crop]
    mid_x = (cropped_x[1:] + cropped_x[:-1])/2.0

    if sqrt:
        diff_y = (np.diff((y - np.min(y))**0.5)/np.mean(np.diff(x)))**2
    else:
        diff_y = np.diff(y - np.min(y))/np.mean(np.diff(x))
    if crop > 0:
        diff_y = diff_y[crop:-crop]
        
    return mid_x, diff_y
        
def find_esf(image, raw_image=None, dx=0.1, blocks=1):
    """Given an oriented image, calculate the PSF from the edge response.

    Arguments:
        image: the image to use for finding the edge, as an NxMx3 RGB array
        raw_image: if a non-demosaiced image is available (as an NxMx3 array), 
            specify it here.  Defaults to using image.
    blocks: how many chunks to split the image into before calculating PSFs for each chunk.
    
    Returns a (blocks, 10*fuzziness-1, 3) array with the PSF in 3 colours.
    """
    if raw_image is None:
        raw_image = image
        bayer_pattern = np.array([[[True]]])
    else:
        bayer_pattern = deduce_bayer_pattern(raw_image)
    
    # Find the step location in each row of the image, then fit a straight line
    #xs, ys = find_edge((image/4).sum(axis=2), fuzziness=fuzziness)
    vertical, falling, line = locate_edge(image) #np.polyfit(xs, ys, 1)
    assert not vertical and not falling, "The edge is the wrong way round!"
    
    try:
        assert np.abs(line[0]*image.shape[0]/blocks) > 1, "The edge is insufficiently slanted."
    except Exception as e:
        #print("y values: ".format(ys))
        #raise e
        print("Warning: {}".format(e))
        print("The edge only moves {} of a pixel over the image blocks".format(line[0]*image.shape[0]/blocks))

    esfs = []
    h = raw_image.shape[0]
    for i in np.arange(blocks):
        # The PSFs are calculated for each "block" of the image
        # We crop blocks out at uniformly spaced heights, centred on the edge
        xslice = slice((i * h)//blocks, ((i+1) * h)//blocks) # the section of the image we're analysing
        xcentre = (xslice.start + xslice.stop)/2.0
        ycentre = xcentre * line[0] + line[1]
        yslice = slice(int(ycentre - 100), int(ycentre + 100))
        print("block at {}, {} has shape {}".format(xcentre, ycentre, raw_image[xslice, yslice, :].shape))
        edge_rgb = []
        for channel in range(3):
            # Align the rows so the transitions are all at x=0
            aligned_rows = extract_aligned_edge(raw_image[xslice, yslice, :], width=100, channel=channel)
            # Then average the rows together using a smoothing spline
            sorted_x, sorted_I, txs = sorted_x_and_I(aligned_rows)
            x, I = average_edge_spline(sorted_x, sorted_I, dx=dx)
            edge_rgb.append((x, I, np.mean(txs)))
        esfs.append(edge_rgb)
    if blocks == 1:
        return esfs[0], line
    else:
        return np.array(esfs), line

def inset_image(fig_or_ax, image, line=None, horizontal=False, flip_line=False):
    """Display a thumbnail of the image, with the edge overlaid as a check."""
    if isinstance(fig_or_ax, matplotlib.axes.Axes):
        ax2 = fig_or_ax
    else:
        ax2 = fig_or_ax.add_axes((0.75, 0.70, 0.15, 0.15), frameon=False)
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

def find_fwhm(x, I, zerolevel=0, annotate_ax=None, color="black", text_y=None):
    y = I
    threshold = (np.max(y) - zerolevel) / 2 + zerolevel
    ileft = np.argmax(y > threshold)
    iright = (len(y) - 1 - np.argmax(y[::-1] > threshold))
    fwhm = x[iright] - x[ileft]
    if annotate_ax is not None:
        annotate_ax.annotate("fwhm {0:.1f}".format(fwhm),
                                xy=(x[ileft], threshold),
                                xytext=(np.min(x), threshold if text_y is None else text_y),
                                arrowprops = dict(facecolor=color, shrink=0),
                                color=color,)
    return fwhm
    
def analyse_file(fname, fuzziness=10, blocks = 11, plot=False, save_plot=False):
    """Analyse one edge image to determine PSF normal to the edge.
    
    fname: string
        the file to be analysed, as an absolute or relative path
    blocks: int (default 11)
        the image is divided into equally-sized blocks along the edge, and the PSF
        is calculated for each block.  Useful when determining field curvature.
    plot: bool (default False)
        if set to True, generate a PDF with plots of the PSF for each block, plus the
        resampled edge image.  If set to an instance of PdfPages, the graph is written
        into that PDF.
    """
    print("Processing {}...".format(fname),end="")
    try:
        bayer_array = load_raw_image(fname)
        rgb_image = (bayer_array.demosaic()//4).astype(np.uint8)
        raw_image = bayer_array.array
    except Exception as e:
        print("Can't load raw data for {}, falling back to JPEG data".format(fname))
        print(e)
        raw_image = None
        rgb_image = imread(fname)
    
    # ensure the images are black-to-white step functions along the second index
    vertical, falling, line = locate_edge(rgb_image)
    rgb_image = reorient_image(rgb_image, vertical, falling)
    raw_image = reorient_image(raw_image, vertical, falling)
    
    # analyse the image to extract sections along the edge
    esfs, line = find_esf(rgb_image, raw_image, blocks=blocks)
    # Each ESF is a length-3 list, containing a tuple for each channel, of:
    #   * x coordinates (with the edge at zero)
    #   * I coordinates (same length as x)
    #   * the mean pixel position, in the original image, of the edge

    psfs = []
    for esf in esfs:
        psf = []
        for x, I, centre_x in esf:
            psf.append((numerical_diff(x, I, sqrt=True, crop=10)))
        psfs.append(psf)

    if plot or save_plot:
        with matplotlib.rc_context(rc={"font.size":6}):
            print(" plotting...",end="")
            fig, ax = plt.subplots(1, 2, figsize=(6,9))
            fig.suptitle(fname)
            
            # Plot the image on the left, and overlay the fitted line
            image_ax = ax[0]
            ys = np.arange(rgb_image.shape[0])
            xs = line[0]*ys + line[1]
            image_ax.imshow(rgb_image[:, int(np.min(xs) - 5*fuzziness):int(np.max(xs) + 5*fuzziness), ...])
            image_ax.xaxis.set_visible(False)
            image_ax.yaxis.set_visible(False)
            image_ax.plot(xs - (np.min(xs) - 5*fuzziness), ys, color="red", dashes=(2,8))

            # Display the PSF for each block of the image
            psf_ax = ax[1]
            psf_max = [0, 0, 0]
            for psf in psfs:
                for channel, (x, I) in enumerate(psf):
                    psf_max[channel] = max(psf_max[channel], np.max(I))

            for i, psf in enumerate(psfs):
                for channel, col, (x, I) in zip(range(3), ['red', 'green', 'blue'], psf):
                    offset =  i*0.66
                    normI = I/psf_max[channel] + offset
                    psf_ax.plot(x, normI, color=col)
                    find_fwhm(x, normI, zerolevel=offset, annotate_ax=psf_ax, color="black", text_y=offset+(channel+1)*0.1)
                centre_y = rgb_image.shape[0]*(i+0.5)/blocks
                image_ax.annotate(str(i), xy=(line[1]+line[0]*centre_y, centre_y))
            fig.tight_layout()
            if save_plot:
                fig.savefig(fname + "_analysis.pdf")
    print(" done.")
    if plot:
        return fig, psfs, esfs
    else:
        return psfs, psfs, esfs


def analyse_files(fnames, output_dir=".", **kwargs):
    """Analyse a number of files.  kwargs passed to analyse_file"""
    spread_functions = {}
    with PdfPages(os.path.join(output_dir, "edge_analysis.pdf")) as pdf:
        for fname in fnames:
            fig, psfs, esfs = analyse_file(fname, plot=True, **kwargs)
            pdf.savefig(fig)
            spread_functions[fname] = {"psfs": psfs, "esfs": esfs}
            plt.close(fig)
    with open(os.path.join(output_dir, "spread_functions.yaml"), 'w') as outfile:
        yaml.dump({f: {"esfs": e, "psfs": p} for f, (e, p) in spread_functions.items()}, outfile)

    return spread_functions
    
def edge_image_fnames(folder):
    """Find all the images in a folder that look like edge images"""
    fnames = []
    for f in os.listdir(folder):
        if f.endswith(".jpg") and "raw" not in f:
            fnames.append(os.path.join(folder, f))
    return fnames

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: {} <file_or_folder> [<file2> ...]".format(sys.argv[0]))
        print("If a file is specified, we produce <file>_analysis.pdf")
        print("If a folder is specified, we produce a single PDF in that folder, analysing all its JPEG contents")
        print("Multiple files may be specified, using wildcards if your OS supports it - e.g. myfolder/calib*.jpg")
        print("In that case, one PDF (./edge_analysis.pdf) is generated for all; the files")
        print("if multiple folders are specified, or a mix of folders and files, each folder is handled separately as above")
        print("and all the files are processed together.")
        print("In the case of multiple files or a folder, we will also save the extracted PSFs in edge_analysis.npz")
        exit(-1)
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        # A single file produces a pdf with a name specific to that file, so it's different.
        analyse_file(sys.argv[1], save_plot=True)
    else:
        fnames = []
        folders = []
        for fname in sys.argv[1:]:
            if os.path.isfile(fname):
                fnames.append(fname)
            if os.path.isdir(fname):
                folders.append(fname)
        if len(fnames) > 0:
            print("Analysing files...")
            analyse_files(fnames)
        for folder in folders:
            #try:
                print("\nAnalysing folder: {}".format(folder))
                analyse_files(edge_image_fnames(folder), output_dir=folder)
            #except Exception as e:
            #    print("ERROR: {}".format(e))
            #    print("Aborting this folder.\n")