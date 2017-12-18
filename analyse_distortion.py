# -*- coding: utf-8 -*-
"""
Analyse a series of step-function images to recover distortion of the image

(c) Richard Bowman 2017, released under GNU GPL

This program expects as input an image containing a single black/white edge.  It will:
1. determine the direction (horizontal/vertical) and sign (black then white or white then black) of the edge
2. track the edge as it is scanned over the field of view
3. analyse the resultant grid of lines to determine the stage's step size and any distortion in the image

NB a 3-channel colour image is assumed.  Pad grayscale images to be n x m x 3 to avoid wierd results...

"""
from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.patches

from skimage import data
import numpy as np
import cv2

import scipy.ndimage
import scipy.ndimage as ndimage
import scipy.interpolate
import itertools
import os
import sys
import re
import os.path
import scipy.optimize
from skimage.io import imread
from matplotlib.backends.backend_pdf import PdfPages

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

def find_edge(image, fuzziness = 5, smooth_x = 5, plot=False):
    """Find the line that best fits an edge

    We reduce images to 2D if they are colour and that the edge is rising and approximately vertical (i.e.
    it goes from low to high as we increase the second array index)"""
    if len(image.shape) > 2:
        image = np.mean(image, axis=2) #ensure grayscale
    #ys = np.zeros(image.shape[0])
    if smooth_x > 0:
        image = scipy.ndimage.gaussian_filter1d(image, sigma=smooth_x, axis=0)
    ys = np.argmax(scipy.ndimage.gaussian_filter1d(image, sigma=fuzziness, order=1, axis=1), axis=1) #find maximum gradient on that line
    xs = np.arange(image.shape[0])
    return xs, ys
    
def noise_on_line(line):
    """Compute how noisy a line is.
    
    We sum the variances of the first derivatives of the X and Y components.  That gives a
    number that will be low (of order 1) if what we have is a horizontal-ish or vertical-ish
    line.
    
    line should be a 2xN numpy array of X and Y positions
    """
    return np.sqrt(np.sum(np.var(np.diff(line, axis=1), axis=1)))
    
def edge_image_fnames(dir):
    """Return a list of all the names of edge image files in a directory"""
    return [n for n in os.listdir(dir) if n.startswith("edge") and n.endswith(".jpg")]
    
def find_edges(dir):
    """Given a directory of edge images, extract the lines and positions from each image."""
    fnames = edge_image_fnames(dir)
    middle_image = imread(os.path.join(dir, fnames[len(fnames)/2]))
    horizontal, falling = find_edge_orientation(middle_image)  # figure out the direction of the edge
    stage_positions = []
    lines = np.empty((len(fnames), 2, middle_image.shape[1 if horizontal else 0]))
    for i, fname in enumerate(fnames):
        print("Processing {}...".format(fname),end="")
        image = imread(os.path.join(dir, fname))
        if horizontal:
            image = image.transpose((1,0,2))  # if the edge is in the first array index, move it to the second
        if falling:
            image = image[:, ::-1, ...]  # for falling edges, flip the image so they're rising
        h, w, d = image.shape
        xs, ys = find_edge(image)
        if horizontal:
            xs, ys = ys, xs
        if falling:
            xs, ys = xs[::-1], ys[::-1]
        lines[i,0,:] = xs
        lines[i,1,:] = ys
        print("done")
    return lines    

def position_from_filename(fname):
    """Extract the x,y,z position from a filename."""
    # extract the stage position from the filename (todo: use EXIF tags or something)
    # the regular expression should return two groups, one is x/y/z the other is a number.
    matches = re.findall(r"([xyz])[_ ]*(-{0,1}[\d]+)", fname)
    pos = dict(matches)
    return np.array([pos[d] for d in ('x','y','z')], dtype=np.int)
    
def find_positions(dir):
    """Given a directory of edge images, extract the lines and positions from each image."""
    stage_positions = []
    for i, fname in enumerate(edge_image_fnames(dir)):
        try:
            stage_positions.append(position_from_filename(fname))
        except:
            print("couldn't extract positions from {}".format(fname))
    return stage_positions
    
def load_edges(dir):
    """Load or calculate the edge positions and shapes from a folder
    
    Returns: liness, positionss
        liness: a list containing two arrays, with the extracted edge shapes
            for vertical and horizontal lines.  Arrays are of shape (n, 2, w) 
            where n is the number of images, 2 represents x/y coordinates, and
            w is the number of pixels across the image (different between the
            two arrays)
        positionss: a list containing two (n, 3) arrays with the stage positions
            corresponding to each edge image.
    """
    liness = [None, None]
    positionss = [None, None]
    for direction_folder in ["distortion_h", "distortion_v"]: # one folder for each edge direction - doesn't matter which
        folder = os.path.join(dir,direction_folder)
        if not os.path.isdir(folder):
            print("Error: folder {} did not exist, skipping.".format(folder))
            continue
        lines_fname = os.path.join(folder,"lines.npz")
        try:
            data = np.load(lines_fname) # cache the image analysis for speed's sake - allows us to repeat the later stuff easily
            lines = data['lines']
            positions = data['stage_positions']
        except:
            print("processing {}...".format(direction_folder))
            lines = find_edges(folder) # this is the slow step of the whole process.
            positions = find_positions(folder)
            np.savez(lines_fname, lines=lines, stage_positions=positions)
            
        changing_axis = np.argmax(np.std(np.mean(lines, axis=2), axis=0)) # the axis that changes is the interesting one - e.g. y for horizontal lines
        liness[changing_axis] = lines
        positionss[changing_axis] = positions
    return liness, positionss
    
    
def reduce_1d(x, blocksize, axis=None):
    """Take block means of a 1D array"""
    if axis is None and len(x.shape) == 1:
        N = len(x)//blocksize
        # We reshape so that we can take block means by taking means of columns.
        return np.mean(x[:(N*blocksize)].reshape(N, blocksize), axis=1)
    else:
        shape = x.shape
        # Crop the array so it reshapes exactly into an integer number of blocks
        N = shape[axis]//blocksize
        crop_slice = ((slice(None),)*(axis)
                      + (slice(0,N*blocksize),) 
                      + (slice(None),)*(len(shape) - axis - 1))
        # then reshape it so the axis in question becomes 2 axes
        new_shape = shape[:axis] + (N, blocksize) + shape[(axis + 1):]
        # the second of these new axes will be the one over which we take the mean
        return np.mean(x[crop_slice].reshape(new_shape), axis=axis+1)

def widen(x, axis=0):
    """widen a boolean array by 1 along the given axis, using a logical OR"""
    shape = list(x.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=np.bool)
    slices = [slice(None)] * len(shape)
    slices[axis] = slice(None, -1)
    out[slices] = x
    slices[axis] = slice(1, None)
    out[slices] = np.logical_or(out[slices], x)
    return out

    
def find_mask_and_deviationss(liness, threshold=3):
    """Filter out noisy bits of the lines, mask the array, and find the deviations-from-means.
    
    returns masked_liness (liness with noisy bits masked off) and 
            deviationss (liness with the mean line shape subtracted)
    """
    masked_liness = [None, None] # "lines" holds all the lines from one folder.  "liness" is a list, holding *both* folders.
    for i in range(len(liness)):
        # calculate the differential variance of each line, smooth it, and set a threshold for "too noisy"
        noisiness = np.sqrt(np.sum(ndimage.filters.gaussian_filter1d(np.diff(liness[i], axis=2)**2, 2, axis=2), axis=1))
        mask = widen(noisiness > threshold, axis=1) # we need to widen the array because the diff() shrank it
        masked_liness[i] = np.ma.masked_array(liness[i], np.tile(mask[:, np.newaxis, :], (1,2,1))) # need to restore axis 1, of length 2

    deviationss = []
    for i, lines in enumerate(masked_liness):
        # First, calculate the mean line shape.  NB we have to subtract each line's position, otherwise masking elements
        # of different lines will add noise (effectively by weighting edge positions differently for different pixels)
        # subtracting the mean of each line should (more or less) fix this by making all the lines coincident before
        # averaging them together.
        # of course, the mean position will be affected by masked elements if lines aren't perfectly straight - but
        # this shouldn't be a big effect as the lines are nearly vertical.  If the mean lines in the plot are straight,
        # that means it's ok - 
        mean_line = (lines - lines.mean(axis=2)[:,:,np.newaxis]).mean(axis=0)
        deviations = lines - mean_line[np.newaxis,...] # remove the average line shape
        deviations -= deviations.mean(axis=2)[:,:,np.newaxis] # remove the positions of individual lines
        deviationss.append(deviations)
    return masked_liness, deviationss
    
def make_dr_spline(dr, liness):
    """Construct a spline to use as a radial distortion function."""
    dr = np.concatenate([[0.], dr]) #fix the centre as zero distortion
    # we model the radial distortion as a function of radius, using cubic interpolation
    # the "x axis" of the function goes from zero to half the diagonal (points beyond are interpolated)
    modelled_deviationss_max_r = np.sqrt(np.sum([np.max(l[:,(i+1)%2,:])**2 
                                                 for i, l in enumerate(liness)]))/2 # half the diagonal
    dr_spline = scipy.interpolate.interp1d(np.linspace(0,modelled_deviationss_max_r,len(dr)),
                                      dr, kind="cubic", bounds_error=False, fill_value="extrapolate")
    def wrapped_dr_spline(radii):
        if isinstance(radii, np.ma.MaskedArray):
            # the spline interpolation fails with a masked array, so unmask-interpolate-remask...
            return np.ma.array(dr_spline(np.ma.filled(radii, 0)), mask=radii.mask)
        else:
            return dr_spline(radii)
            
    return wrapped_dr_spline
    
def modelled_deviationss(centre, dr, liness):
    """Calculate the deviation-from-the-mean lines for a given distortion function
    
    centre: np.array, length 2, describing the centre of distortion in pixels
    """
    deviationss = []
    dr_spline = make_dr_spline(dr, liness)
    
    centre = np.array(centre)
    for i, lines in enumerate(liness):
        pos = lines - centre[np.newaxis,:,np.newaxis] #positions relative to centre
        radii = np.sqrt(np.sum(pos**2, axis=1)) #radii, for each point on each line
        dr = dr_spline(radii)
        deviations = dr[:,np.newaxis,:] * pos/radii[:,np.newaxis,:] # NB this is the "true" deviation in r, not what we measure
        deviations[:,(i + 1) % 2, :] = 0 # We can't measure anything along the axis of the deviation, so zero it out
                                         # NB there's an approximation here because the edge isn't perfectly along x or y...
        deviationss.append(deviations)
    return deviationss

def plot_lines(ax, lines, deviations=None, reduction=1):
    """Plot lines, first making all the lines the same shape, then adding some deviation to them.
    
    ax is the matplotlib Axes into which we draw the lines
    lines should be an Nx2xM array, with N lines each having M points on them.
    deviations should have the same shape as lines, if specified
    reduction does block averaging on the lines to remove noise and speed up plotting.
    """
    line_centres = np.mean(lines, axis=2)
    mean_line = np.mean(lines - line_centres[:,:,np.newaxis], axis=0)
    if deviations is None:
        # if deviations is none, set it so we plot the original lines...
        deviations = lines - (line_centres[:,:,np.newaxis] + mean_line[np.newaxis,:,:])
    for i in range(lines.shape[0]):
        ax.plot(reduce_1d(line_centres[i,0] + mean_line[0,:] + deviations[i,0,:], reduction),
                reduce_1d(line_centres[i,1] + mean_line[1,:] + deviations[i,1,:], reduction))
    
def tidy_pixel_axes(axes, 
                    label_pos=(0.03,0.97), 
                    va="top", ha="left", 
                    xlabels=None, ylabels=[0], 
                    aspect=1, 
                    part_number_offset=0,
                    xlabel="x position (pixels)", ylabel="y position(pixels)"):
    """For a series of axes with shared X and Y, ensure the aspects are equal, limits are tight, and label them.
    
    label_pos should be a tuple of two numbers setting the position of (a), (b), ... "part labels" in axis coords.
    va and ha set the alignment of the above labels
    xlabels and ylabels should be lists of integers specifying which axes have x axis and y axis labels.  Leave as
        None to label the middle axis, or True to label all axes
    aspect (if not None) sets the aspect of the plot - defaults to 1
    part_number_offset starts the figure part labels from later in the sequence - e.g. if it's 3, then we start at (d).
    
    This works very nicely with plot_lines...
    """
    for i, ax in enumerate(axes):
        ax.set_adjustable('box-forced') # avoid extraneous whitespace (https://github.com/matplotlib/matplotlib/issues/1789/)
        if aspect is not None:
            ax.set_aspect(aspect) # X and Y are both in pixels - so make sure it's isotropic.
        if label_pos is not False:
            ax.text(label_pos[0], label_pos[1], "(" + chr(i+97+part_number_offset) + ")", va=va, ha=ha, transform=ax.transAxes)
    if xlabels is None:
        xlabels = [len(axes)//2]
    if xlabels is True:
        xlabels = range(len(axes))
    if ylabels is True:
        ylabels = range(len(axes))
    if xlabel is not None:
        for i in xlabels:
            axes[i].set_xlabel(xlabel)
    if ylabel is not None:
        for i in ylabels:
            axes[i].set_ylabel(ylabel)
   
    
def analyse_distortion(dir):
    """Analyse folders full of horizontal and vertical edge images for distortion"""
    liness, positionss = load_edges(dir)
    liness, deviationss = find_mask_and_deviationss(liness)
    
    # Fit the distortion with a purely radial function
    reduction = 50 # Work at reduced resolution, to decrease noise and increase speed...
    reduced_liness, reduced_deviationss = [], []
    for lines, deviations in zip(liness, deviationss):
        reduced_liness.append(reduce_1d(lines, 50, axis=2))
        reduced_deviationss.append(reduce_1d(deviations, 50, axis=2))

    def cost_function(coeffs, centre=None):
        """Return the error between calculated and actual deviations.  
        
        The first two elements of coeffs are interpreted as the centre if it's not passed explicitly.
        """
        if centre is None:
            mdevss = modelled_deviationss(coeffs[:2], coeffs[2:], reduced_liness)
        else:
            mdevss = modelled_deviationss(centre, coeffs, reduced_liness)
        cost = 0
        for devs, mdevs in zip(reduced_deviationss, mdevss):
        # NB we're not sensitive to distortion that appears to move the whole line, so we
        # subtract the mean of each line, to get rid of shifts like that.  Otherwise, it acts
        # as a built-in penalty for large distortions.
            cost += np.var((mdevs - devs) - np.mean(mdevs - devs, axis=2)[:,:,np.newaxis])
        return cost
        
    camera_centre = [(l.shape[2]-1)/2 for l in reversed(liness)] # this is the centre of the camera
    res = scipy.optimize.minimize(cost_function, np.array(camera_centre + [0]*4)) # actually run the optimisation...
    
    figures = []
    matplotlib.rcParams.update({'font.size':6})
    # Plot the lines as extracted from the images
    f, axes = plt.subplots(1,4, sharex=True, sharey=True, figsize=(8,2.5), )
    figures.append(f)
    gain = 50
    reduction = 50
    axes[0].set_title("Lines as found on the images")
    axes[1].set_title("Deviations magnified {}x".format(gain))
    axes[2].set_title("With modelled deviations")
    axes[3].set_title("Residual deviations")
    md = modelled_deviationss(res.x[:2], res.x[2:], liness)
    for lines, dev, mod in zip(liness, deviationss, md):
        plot_lines(axes[0], lines, reduction=reduction)
        plot_lines(axes[1], lines, dev*gain, reduction=reduction)
        plot_lines(axes[2], lines, mod*gain, reduction=reduction)
        plot_lines(axes[3], lines, (dev-mod)*gain, reduction=reduction)
    tidy_pixel_axes(axes)
    
    # Plot the radial deviations, along with the model
    fig, axes = plt.subplots(1,2, figsize=(8,3), sharex=True, sharey=True)
    figures.append(fig)
    dr_spline = make_dr_spline(np.concatenate([[0.], res.x[2:]]), liness)
    centre = np.array(res.x[:2])
    for i, (lines, deviations) in enumerate(zip(liness, deviationss)):
        pos = lines - centre[np.newaxis,:,np.newaxis] # positions relative to centre
        radii = np.sqrt(np.sum(pos**2, axis=1)) # radii, for each point on each line
        cosines = pos[:,i,:] / radii # the deviation we measure is out by a factor cos(angle)
        dr = deviations[:,i,:] / cosines
        # we can't measure shift of the whole line, so find the mean shift of the model and add it.
        # NB the model calculates dr, so first convert to dy, then take the mean, then back to dr
        dr += np.mean(dr_spline(radii) * cosines) / cosines
        for j in range(radii.shape[0]):
            axes[0].plot(reduce_1d(radii[j,:], 50), reduce_1d(dr[j,:], 50), '.', markersize=2.0)
            axes[1].plot(reduce_1d(radii[j,:], 50), reduce_1d(dr[j,:] - dr_spline(radii[j,:]), 50), '.', markersize=2.0)
    modelled_deviationss_max_r = np.sqrt(np.sum([l.shape[2]**2 for l in liness]))/2
    radii = np.linspace(0, modelled_deviationss_max_r, 200)
    axes[0].plot(radii, dr_spline(radii), '-', linewidth=2, color="black")
    axes[1].plot(radii, np.zeros_like(radii), '-', linewidth=2, color="black")
    for ax in axes:
        ax.set_ylim((-10,10))
        ax.set_xlabel("Radial position (pixels)")
    axes[0].set_ylabel("Radial distortion (pixels)")
    axes[0].set_title("Radial distortion, with fit")
    axes[1].set_title("Residuals")

    # plot distortion as images
    fig, axes = plt.subplots(2,3, figsize=(8,8))
    figures.append(fig)
    reduction = 100 # it makes sense to average pixels together quite aggressively
    md = modelled_deviationss(res.x[:2], res.x[2:], liness)
    for i, (deviations, modelled) in enumerate(zip(deviationss, md)):
        rdev = reduce_1d(deviations, reduction, axis=2)[:,i,:]
        rmdev = reduce_1d(modelled, reduction, axis=2)[:,i,:]
        rmdev -= np.mean(rmdev, axis=1)[:,np.newaxis] # get rid of whole-line shifts, as we can't measure them anyway
        vrange = np.percentile(np.abs(rdev), 95)
        axes[i,0].imshow(rdev, aspect="auto", vmin=-vrange, vmax=vrange, cmap="PuOr")
        axes[i,1].imshow(rmdev, aspect="auto", vmin=-vrange, vmax=vrange, cmap="PuOr")
        im = axes[i,2].imshow(rdev-rmdev, aspect="auto", vmin=-vrange, vmax=vrange, cmap="PuOr")
        fig.colorbar(im, ax=axes[i,2])
        tidy_pixel_axes(axes[i,:], aspect=None, xlabel="pixel position (/{})".format(reduction), 
                        ylabel="edge position", part_number_offset=i*len(axes))

    # plot stage vs image position (to calibrate the stage)
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    figures.append(fig)
    for i, (lines, positions) in enumerate(zip(liness, positionss)):
        stage_changing_axis = np.argmax(np.std(positions, axis=0)) # find which direction the *stage* is moving in
        stage_y = positions[:,stage_changing_axis]
        camera_y = np.mean(lines[:,i,:], axis=1)
        m = np.polyfit(stage_y, camera_y, 1)
        axes[i].plot(stage_y, camera_y, 'k+')
        axes[i].plot(stage_y, stage_y*m[0]+m[1], 'b-')
        axes[i].text(0.5, 0.01, '{:.2f}px/step'.format(np.abs(m[0])),
            verticalalignment='bottom', horizontalalignment='center', transform=axes[i].transAxes)
        axes[i].set_title("stage vs pixels" + ["(h)","(v)"][i])
    
    return figures
    
def analyse_dir(dir, summary_pdfs={}):
    """Analyse a folder of edge images for distortion.
    
    Results are saved in a PDF in the folder.
    
    If summary_pdfs is specified, it should be a dictionary, with numeric keys,
    of PdfPages objects.  Figure n from each folder will be saved into 
    summary_pdfs[n].
    """
    print("Analysing directory {}".format(dir))
    with PdfPages(os.path.join(dir, "distortion_analysis.pdf")) as pdf:
        figs = analyse_distortion(dir)
        for f in figs:
            f.suptitle("Objective "+os.path.basename(dir))
        for i, f in enumerate(figs):
            pdf.savefig(f)
            try: summary_pdfs[i].savefig(f)
            except: pass
            plt.close(f)
    
def analyse_dirs(dirs):
    """Analyse a series of directories, generating summary PDFs as well as individual ones."""
    with PdfPages("distortion_summary_grids.pdf") as summary_grids, \
        PdfPages("distortion_summary_radial.pdf") as summary_radial, \
        PdfPages("distortion_summary_images.pdf") as summary_images:
        for dir in dirs:
            analyse_dir(dir, summary_pdfs={0:summary_grids, 1:summary_radial, 2:summary_images})

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: {} <folder> [<folder> ...]".format(sys.argv[0]))
        print("For each folder specified, look for subfolders 'distortion_h' and 'distortion_v'")
        print("These folders should contain images of an edge, scanned across the image perpendicular")
        print("to the edge.  This program fits the edge, and looks for distortion as we move over the")
        print("field of view.")
        print("It outputs a PDF of plots, 'distortion_analysis.pdf', for each folder analysed.")
        print("If more than one folder is specified, it generates a number of summary PDF files too.")
        exit(-1)
    if len(sys.argv) == 2:
        analyse_dir(sys.argv[1])
    else:
        analyse_dirs(sys.argv[1:])