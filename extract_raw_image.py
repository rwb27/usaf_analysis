from __future__ import print_function
#import microscope
#import picamera
import numpy as np
import matplotlib.pyplot as plt
import time
import picamera_array
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import PIL.ExifTags
from dump_exif import exif_data_as_string
import sys

full_resolution=(3280,2464)

class DummyCam(object):
    resolution = full_resolution
    revision = 'IMX219'
    sensor_mode = 0
    
def load_raw_image(filename, ArrayType=picamera_array.PiBayerArray, open_jpeg=False):
    with open(filename, mode="rb") as file:
        jpeg = file.read()
    cam = DummyCam()
    bayer_array = ArrayType(cam)
    bayer_array.write(jpeg)
    bayer_array.flush()
    
    if open_jpeg:
        jpeg = PIL.Image.open(filename)
        # with thanks to https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
        exif_data = jpeg._getexif()
        return bayer_array, jpeg, exif_data
    return bayer_array
    
def extract_file(filename):
    """Extract metadata and raw image from a file"""
    print("converting {}...".format(filename))
    bayer_array, jpeg, exif_data = load_raw_image(filename, open_jpeg=True)
    
    # extract EXIF metadata from the image
    root_fname, junk = filename.rsplit(".j", 2) #get rid of the .jpeg extension
    with open(root_fname + "_exif.txt", "w") as f:
        f.write(exif_data_as_string(jpeg))
    
    # extract raw bayer data
    cv2.imwrite(root_fname + "_raw.tif", bayer_array.demosaic())

if __name__ == "__main__":
    for filename in sys.argv[1:]:
        extract_file(filename)