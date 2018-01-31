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
import matplotlib
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
import itertools
import os
import sys
import re
import os.path
from skimage.io import imread
from matplotlib.backends.backend_pdf import PdfPages
from analyse_distortion import find_edge_orientation, find_edge, reduce_1d
from extract_raw_image import load_raw_image
import analyse_distortion

#################### Rotate the image so the bars are X/Y aligned #############

def reorient_image(image, horizontal, falling):
    """Flip or transpose an image.
    
    image: a 3D array representing a (colour) image
    horizontal: whether the image should be transposed
    falling: whether the image should be mirrored
    
    Returns: image, flipped/transposed as requested.
    """
    if not horizontal:
        image = image.transpose((1,0,2))  # if the edge is in the first array index, move it to the second
    if falling:
        image = image[:, ::-1, ...]  # for falling edges, flip the image so they're rising
    return image
    
def locate_edge(image, fuzz=10):
    """Fit the peak gradients of the image with a straight line.
    
    Returns: (horizontal, line)
        horizontal: bool
            Whether the line is closer to horizontal than vertical.
        line: tuple of two floats (gradient, intercept)
            The equation of the line.  If "horizontal", y=gradient*x + intercept.
            If horizontal is false, then x=y*gradient + intercept.  x and y refer
            to the first and second array indices of the image respectively.
    """
    gray_image = image.mean(axis=2) if len(image.shape) == 3 else image
    # start with a non-directional edge detection
    from scipy.ndimage.filters import gaussian_filter
    edges = gaussian_filter(gray_image, fuzz, order=(0,1))**2
    edges += gaussian_filter(gray_image, fuzz, order=(1,0))**2
    # background-subtract to remove noise
    edges -= edges.max()*0.3
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
    if Sxx > Syy: # the line is ~horizontal
        gradient = Sxy/Sxx
        intercept = cy - cx * gradient
        return True, (gradient, intercept)
    else: # we have a vertical-ish line
        gradient = Sxy/Syy
        intercept = cx - cy * gradient
        return false, (gradient, intercept)

        
def edge_is_falling(image, line):
    """Determine if an edge is white-to-black."""
    gray_image = image.mean(axis=2) if len(image.shape) == 3 else image
    return np.sum(gray_image[y < x*gradient + intercept]) > 0.5*np.sum(gray_image)
        
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
    
def average_edge(image, line, subsampling=10, bayer_pattern=np.array([[[True]]])):
    """Average along an edge, returning a subsampled marginal distribution.
    
    image: the edge image to average.  Should be a vertical black-white edge.
        This should be a 3D numpy array - though the third dim may be length 1
    line: the coefficients of a straight line describing the edge - 2 elements
    subsampling: how many bins each pixel is divided into.
    bayer_pattern: 3D boolean array representing the Bayer pattern.  Default
        value will assume every pixel has valid values for every colour channel.
        
    Returns:
        subsampled_edge: a 2D array with the edge function for each colour.
        
    ==============
    Bayer patterns
    ==============
    * If bayer_pattern is None, we assume the camera has true RGB pixels.
    * If bayer_pattern is a scalar, it should be the side length of the unit
    cell of the Bayer pattern.  
    """
    xs = np.arange(image.shape[0]) # for convenience, X values of the image
    
    # We need to make the average wider than one line to allow shifting
    margin = subsampling * int(xs.shape[0]*np.abs(line[0]))//2 + 4*subsampling
    # To take the average, we need the sum and the number of terms:
    total_intensity = np.zeros((image.shape[1]*subsampling + 2*margin, image.shape[2]))
    total_n = np.zeros_like(total_intensity)
    bw, bh, n_channels = bayer_pattern.shape
    # Repeat the bayer pattern to make up a whole column (NB it's >1 row wide)
    bayer_col = np.tile(bayer_pattern, (1,image.shape[1]//bayer_pattern.shape[1]+1,1))
    bayer_col = bayer_col[:, :image.shape[1], :] # In case the image is not an integer number of cells
    
    # Calculate the shift for each line in the image
    y_shifts = -(xs - xs.shape[0]/2)*line[0] # Use the fitted line
    dys = np.round(y_shifts*subsampling).astype(int)
    for x, dy in zip(xs, dys):
        #y_shift = float(dy)/subsampling
        #ax.plot(np.arange(image.shape[1]/2)*2 + y_shift, image[x,(x % 2)::2,1])
        rr = slice(margin+dy, -margin+dy) # this aligns our row with the others
        total_intensity[rr,:] += np.repeat(image[x,:,:], subsampling, axis=0)
        total_n[rr] += np.repeat(bayer_col[x % bw, :, :], subsampling, axis=0)
    return total_intensity[margin:-margin, :] / total_n[margin:-margin, :]

    
def find_esf(image, raw_image=None, fuzziness=10, subsampling=10, blocks=1):
    """Given an oriented image, calculate the PSF from the edge response.

    image: the image to use for finding the edge, etc. - should be debayered.
    raw_image: if a non-debayered image is available, specify it here.  Defaults to using image.
    fuzziness: sets the smoothing parameter - in general, should be ~ the expected resolution in pixels.
    subsampling: how much sub-pixel sampling to do (10 is reasonable)
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
    line = locate_edge(image) #np.polyfit(xs, ys, 1)
    
    if subsampling > 1: # Check that the edge is slanted sufficiently to use subsampling
        try:
            assert np.abs(line[0]*image.shape[0]/blocks) > 4, "Error: to use subsampling you must have a slanted " \
                                                "edge, {} is too straight for image {}!".format(line, image.shape)
        except Exception as e:
            #print("y values: ".format(ys))
            raise e

    esfs = []
    h = raw_image.shape[0]
    def round_to_bw(x):
        """Round a number to an integer multiple of the Bayer pattern width"""
        return int(np.round(x/bayer_pattern.shape[1])) * bayer_pattern.shape[1]
    for i in np.arange(blocks):
        # The PSFs are calculated for each "block" of the image
        # We crop blocks out at uniformly spaced heights, centred on the edge
        xslice = slice((i * h)//blocks, ((i+1) * h)//blocks) # the section of the image we're analysing
        xcentre = (xslice.start + xslice.stop)/2.0
        ycentre = xcentre * line[0] + line[1]
        yslice = slice(round_to_bw(ycentre - 5*fuzziness), round_to_bw(ycentre + 5*fuzziness))
        print("block at {}, {} has shape {}".format(xcentre, ycentre, raw_image[xslice, yslice, :].shape))
        esfs.append(average_edge(raw_image[xslice, yslice, :], line, subsampling=subsampling, bayer_pattern=bayer_pattern))
    if blocks == 1:
        return esfs[0], line
    else:
        return np.array(esfs), line

def plot_psf(psf, ax=None, x=None, xlabel="position/pixels", subsampling=1):
    if ax is None:
        f, ax = plt.subplots(1,1)
    if x is None:
        x = np.arange(psf.shape[0])/subsampling
    for i, col in enumerate(['red', 'green', 'blue']):
        ax.plot(x, psf[:, i], color=col)
    ax.set_xlabel(xlabel)

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
    
def analyse_file(fname, fuzziness=10, subsampling = 10, blocks = 11, plot=False, save_plot=False):
    """Analyse one edge image to determine PSF normal to the edge.
    
    fname: string
        the file to be analysed, as an absolute or relative path
    subsampling: int (default 1)
        whether to up-sample the image to get a smoother PSF - 1 means no subsampling
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
    horizontal, falling = find_edge_orientation(rgb_image)
    rgb_image = reorient_image(rgb_image, horizontal, falling)
    raw_image = reorient_image(raw_image, horizontal, falling)
    
    # analyse the image to extract sections along the edge
    esfs, line = find_esf(rgb_image, raw_image, fuzziness=fuzziness, subsampling=subsampling, blocks=blocks)
    psfs = scipy.ndimage.gaussian_filter1d(esfs, subsampling/2, order=1, axis=1, mode="nearest")

    if plot or save_plot:
        with matplotlib.rc_context(rc={"font.size":6}):
            print(" plotting...",end="")
            fig = plt.figure(figsize=(12,9), )
            fig.suptitle(fname)
            nrows = int(np.floor(np.sqrt(blocks + 1)))
            ncols = int(np.ceil(float(blocks + 1)/nrows))
            gs = GridSpec(nrows, ncols + 1) # we'll plot things in a grid.
            
            image_ax = fig.add_subplot(gs[:,0])
            ys = np.arange(rgb_image.shape[0])
            xs = line[0]*ys + line[1]
            image_ax.imshow(rgb_image[:, int(np.min(xs) - 5*fuzziness):int(np.max(xs) + 5*fuzziness), ...])
            image_ax.xaxis.set_visible(False)
            image_ax.yaxis.set_visible(False)
            image_ax.plot(xs - (np.min(xs) - 5*fuzziness), ys, color="red", dashes=(2,8))
            for i in range(blocks):
                ax = fig.add_subplot(gs[i//ncols, 1 + (i % ncols)])
                plot_psf(psfs[i,...], ax=ax, subsampling=subsampling)
                ax.set_title("PSF {}".format(i))
                find_fwhm(psfs[i,...], annotate_ax=ax, subsampling=subsampling)
                centre_y = rgb_image.shape[0]*(i+0.5)/blocks
                image_ax.annotate(str(i), xy=(line[1]+line[0]*centre_y, centre_y))
            fig.tight_layout()
            if save_plot:
                fig.savefig(fname + "_analysis.pdf")
    print(" done.")
    if plot:
        return fig, psfs, esfs, subsampling
    else:
        return psfs, psfs, esfs, subsampling


def analyse_files(fnames, output_dir=".", **kwargs):
    """Analyse a number of files.  kwargs passed to analyse_file"""
    psf_list = []
    esf_list = []
    with PdfPages(os.path.join(output_dir, "edge_analysis.pdf")) as pdf:
        for fname in fnames:
            fig, psfs, esfs, subsampling = analyse_file(fname, plot=True, **kwargs)
            pdf.savefig(fig)
            psf_list.append(psfs)
            esf_list.append(esfs)
            plt.close(fig)
    np.savez(os.path.join(output_dir, "edge_analysis.npz"), filenames=fnames, psfs=psf_list, esfs=esf_list, subsampling=subsampling)
    return np.array(psf_list), np.array(esf_list), subsampling
    
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
            try:
                print("\nAnalysing folder: {}".format(folder))
                analyse_files(edge_image_fnames(folder), output_dir=folder)
            except Exception as e:
                print("ERROR: {}".format(e))
                print("Aborting this folder.\n")