# -*- coding: utf-8 -*-
"""
Analyse a folder of z-stacks of edge images, to recover resolution and field curvature.

"""
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import cv2

import scipy.ndimage
import scipy.interpolate
import itertools
import os
import sys
import re
import os.path
from matplotlib.backends.backend_pdf import PdfPages
import analyse_edge_image
import analyse_distortion

def cached_psfs(folder, fnames):
    """Attempt to retrieve the previously-analysed PSF/ESF from a file"""
    npz = np.load(os.path.join(folder, "edge_analysis.npz"))
    cached_fnames = npz['filenames']
    cached_psfs = npz['psfs']
    cached_esfs = npz['esfs']
    psfs = np.zeros((len(fnames), ) + cached_psfs.shape[1:], dtype=cached_psfs.dtype)
    esfs = np.zeros((len(fnames), ) + cached_esfs.shape[1:], dtype=cached_esfs.dtype)
    for i, fname in enumerate(fnames):
        hit = False
        for j, cached_fname in enumerate(cached_fnames):
            if cached_fname.endswith(fname):
                psfs[i,...] = cached_psfs[j,...]
                esfs[i,...] = cached_esfs[j,...]
                hit = True
                break
        assert hit, "Couldn't find a match for {}".format(fname)
    return psfs, esfs, npz['subsampling']

def analyse_zstack(folder):
    """Find the point spread function of each image in a Z stack series"""
    # The folder usually contains raw and non-raw versions - use the raw ones if possible
    fnames = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    assert len(fnames) > 0, "There were no files in the folder '{}'".format(folder)
    if len([f for f in fnames if "raw" in f]) > 0:
        fnames = [f for f in fnames if "raw" in f]
    # Extract the stage position from the filenames
    positions = np.array([analyse_distortion.position_from_filename(f) for f in fnames])
    print(positions)
    assert np.all(np.var(positions, axis=0)[:2] == 0), "Only the Z position should change!"
    # extract the PSFs
    try:
        raise Exception("I don't want to use the cache.")
        psfs, esfs, subsampling = cached_psfs(folder, fnames)
    except:
        print("Couldn't find cached PSFs, analysing images (may take some time...")
        psfs, esfs, subsampling = analyse_edge_image.analyse_files([os.path.join(folder, f) for f in fnames], output_dir=folder)
    # plot the PSFs as YZ slices
    blocks = psfs.shape[1]
    zs = positions[:,2]
    z_bounds = np.concatenate(([zs[0]], (zs[1:] + zs[:-1])/2., [zs[-1]])) # pcolormesh wants edge coords not centre coords
    y_bounds = np.arange(psfs.shape[2] + 1) - psfs.shape[2]/2.0
    fig, axes = plt.subplots(1,blocks, figsize=(1*blocks, 2), sharey=True)
    psf8 = (psfs/np.max(psfs)*255).astype(np.uint8)
    psf8[psfs<0] = 0
    for i in range(blocks):
        #axes[i].pcolormesh(y_bounds, z_bounds, psf8[:,i,...], aspect="auto")
        axes[i].imshow(psf8[:,i,...], aspect="auto")
    # calculate sharpness metrics and fit for field curvature
    ys = np.arange(psfs.shape[2])[np.newaxis,np.newaxis,:,np.newaxis]
    mean_ys = np.mean(np.abs(psfs)*ys, axis=2)/np.mean(np.abs(psfs), axis=2)
    var_ys = np.mean(np.abs(psfs)*(ys-mean_ys[:,:,np.newaxis,:])**2, axis=2)/np.mean(np.abs(psfs), axis=2)
    field_curvature = np.argmin(var_ys, axis=0) # todo: use the thresholded centre of mass code from USAF
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(field_curvature)
    with PdfPages(os.path.join(folder, "edge_zstack_summary.pdf")) as pdf:
        for f in [fig, fig2]:
            pdf.savefig(f)
            plt.close(f)
    
    
    
    

if __name__ == "__main__":
    try:
        path = sys.argv[1]
        assert os.path.isdir(path)
    except:
        print("Usage: {} <folder> [<folder> ...]".format(sys.argv[0]))
        print("This script expects arguments that are folders, containing jpeg files of edge images.")
        print("This scripts creates a number of PDF plots based on said images, per folder.")
    
    for dir in sys.argv[1:]:
        analyse_zstack(dir)