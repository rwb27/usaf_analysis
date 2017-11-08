# A test script for analysing distortion, debugging the main one
from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.patches

from skimage import data
import numpy as np
import cv2
from cv2 import imread

import scipy.ndimage
import scipy.interpolate

from analyse_distortion import *

def plot_image_with_line(fname, ax):
    """Fit a line to the edge in one image, specified by filename"""
    ax.clear()
    image = imread(fname)
    ax.imshow(image[::-1,:,...])
    
    horizontal, falling = find_edge_orientation(image)
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
    ax.plot(ys, xs, "r-")
    

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    dir = os.path.join(here, "datasets", "pilens_2_ss", "distortion_h")
    fnames = [os.path.join(dir, fn) for fn in edge_image_fnames(dir)]

    if False:
        f, ax = plt.subplots(1,1)
        current_image = 0
        plot_image_with_line(fnames[current_image], ax)
        
        def flip_through_images(event):
            global current_image
            if event.key == '.':
                current_image += 5
            if event.key == ',':
                current_image -= 1
            if current_image in range(len(fnames)):
                print("showing image {}".format(current_image))
                plot_image_with_line(fnames[current_image], ax)
                f.canvas.draw()
        f.canvas.mpl_connect('key_press_event', flip_through_images)
        plt.show()
    
    if True:
        image = imread(fnames[-1])
        fuzziness=5
        smooth_x=5
        edge_image = scipy.ndimage.median_filter(np.mean(image,axis=2), size=10)
        edge_image = (scipy.ndimage.gaussian_filter1d(edge_image, sigma=fuzziness, order=1, axis=1)**2 +              scipy.ndimage.gaussian_filter1d(edge_image, sigma=fuzziness, order=1, axis=0)**2 )
        edge_image = edge_image/edge_image.max()
        
        f, axes = plt.subplots(1,2)

        
            
        plt.show()