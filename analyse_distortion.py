# -*- coding: utf-8 -*-
"""
Analyse a series of step-function images to recover distortion of the image

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
import numpy as np
import cv2

import scipy.ndimage
import scipy.interpolate
import itertools
import os
import sys
import re
import os.path
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

def analyse_distortion(dir):
    """Analyse folders full of horizontal and vertical edge images for distortion"""
    f, axes = plt.subplots(1,3)
    axes[0].set_title("Lines as found on the images")
    axes[1].set_title("Horizontal lines (- mean y)")
    axes[2].set_title("Vertical lines (- mean x)")
    f2, ax2 = plt.subplots(1,2)
    ax2[0].set_title("Deviation from linearity (h)")
    ax2[1].set_title("Deviation from linearity (v)")
    f3, ax3 = plt.subplots(1,2)
    ax3[0].set_title("Deviation from mean (h)")
    ax3[1].set_title("Deviation from mean (v)")
    f4, ax4 = plt.subplots(1,2)
    ax4[0].set_title("Deviation from mean (h)")
    ax4[1].set_title("Deviation from mean (v)")
    f5, ax5 = plt.subplots(1,1)
    ax5.set_title("Deviation from mean (radial)")
    f6, ax6 = plt.subplots(1,2)
    ax6[0].set_title("stage vs pixels (h)")
    ax6[0].set_title("stage vs pixels (v)")
    f7, ax7 = plt.subplots(1,3)
    for i, m in enumerate([5,10,50]):
        ax7[i].set_title("Distortion multiplied {}x".format(m))
    max_r = 0
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
        max_changing_axis = np.max(lines[:,changing_axis, :])
        
        blocksize = 50
        deviation_from_linearity = np.zeros((lines.shape[0], lines.shape[2]//blocksize))
        deviation_from_mean = np.zeros_like(deviation_from_linearity)
        clean_lines = np.zeros(lines.shape[0])
        for i in range(lines.shape[0]):
            if noise_on_line(lines[i,...]) < 10:
                clean_lines[i] = 1
        mean_line = np.mean(lines*clean_lines[:,np.newaxis,np.newaxis], axis=0)/np.mean(clean_lines) # the mean shape should ignore bad lines.
        mean_y = np.zeros(lines.shape[0])
        for i in range(lines.shape[0]):
            # Perform a linear fit, and subtract the fitted line, to get deviation from linearity
            ys = lines[i,changing_axis,:].copy()
            mean_y[i] = np.mean(ys)
            xs = lines[i,(changing_axis + 1) % 2,:]
            m = np.polyfit(xs, ys, 1, w=np.logical_and(ys > 100, ys < max_changing_axis)) #fit a straight line
            delin_ys = ys - m[0]*xs - m[1]
            demean_ys = ys - mean_line[changing_axis,:]
            demean_ys -= np.mean(demean_ys)
            assert np.all(demean_ys.shape == ys.shape)
            # take a block average of the line, after detrending, to use for the image plot
            deviation_from_linearity[i,:] = delin_ys[:deviation_from_linearity.shape[1]*blocksize].reshape(deviation_from_linearity.shape[1], blocksize).mean(axis=1)
            deviation_from_mean[i,:] = demean_ys[:deviation_from_mean.shape[1]*blocksize].reshape(deviation_from_linearity.shape[1], blocksize).mean(axis=1)
            if clean_lines[i]>0: #some images may not have an edge(e.g. if it was off-screen)
                axes[0].plot(lines[i,0,:], lines[i,1,:])
                for i, m in enumerate([5, 10, 50]):
                    ex = xs
                    ey = ys + demean_ys * m
                    if changing_axis == 0:
                        ex, ey = ey, ex
                    ax7[i].plot(ex, ey)
                ax3[changing_axis].plot(scipy.ndimage.gaussian_filter(demean_ys, sigma=20))
                axes[changing_axis+1].plot(scipy.ndimage.gaussian_filter(delin_ys, sigma=20))
                cx, cy = (1640, 1232) if changing_axis ==0 else (1232, 1640)
                rs = np.sqrt((xs-cx)**2 + (ys-cy)**2)
                max_r = max(max_r, rs.max())
                ax5.plot(rs, scipy.ndimage.gaussian_filter(demean_ys, sigma=20) * rs/ys, linewidth=0.5)
        vrange = np.percentile(np.abs(deviation_from_linearity * clean_lines[:,np.newaxis]), 95) # there may be some noisy images - using centiles is robust to these.
        image = ax2[changing_axis].imshow(deviation_from_linearity, aspect="auto", vmin=-vrange, vmax=vrange, cmap="PuOr")
        f2.colorbar(image)
        vrange = np.percentile(np.abs(deviation_from_mean * clean_lines[:,np.newaxis]), 95)
        image = ax4[changing_axis].imshow(deviation_from_mean, aspect="auto", vmin=-vrange, vmax=vrange, cmap="PuOr")
        f4.colorbar(image)
        # plot pixels vs stage position
        stage_changing_axis = np.argmax(np.std(positions, axis=0)) # find which direction the *stage* is moving in
        stage_y = positions[:,stage_changing_axis]
        m = np.polyfit(stage_y, mean_y, 1, w=clean_lines)
        ax6[changing_axis].plot(stage_y, mean_y, 'k+')
        ax6[changing_axis].plot(stage_y, stage_y*m[0]+m[1], 'b-')
        ax6[changing_axis].text(0.5, 0.01, '{:.2f}px/step'.format(np.abs(m[0])),
            verticalalignment='bottom', horizontalalignment='center', transform=ax6[changing_axis].transAxes)
    axes[0].set_aspect(1)
    axes[0].set_xlabel("Position in image (pixels)")
    axes[0].set_ylabel("Position in image (pixels)")
    for i in range(3):
        ax7[i].set_aspect(1)
        ax7[i].set_xlim(axes[0].get_xlim())
        ax7[i].set_ylim(axes[0].get_ylim())
    ax5.set_ylim((-5,5))
    ax5.plot([0,max_r],[0,0],'k-', linewidth=2)
    return((f, f2, f3, f4, f5, f6, f7))
    
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
    with PdfPages("distortion_summary_radial.pdf") as summary_radial, \
        PdfPages("distortion_summary_images.pdf") as summary_images, \
        PdfPages("distortion_summary_grids.pdf") as summary_grids:
        for dir in dirs:
            analyse_dir(dir, summary_pdfs={4:summary_radial, 0:summary_grids, 3:summary_images})

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