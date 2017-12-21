from __future__ import print_function
#import microscope
#import picamera
import numpy as np
#import matplotlib.pyplot as plt
import time
import picamera_array
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
    
def load_raw_image(filename, ArrayType=picamera_array.PiSharpBayerArray, open_jpeg=False):
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
    cv2.imwrite(root_fname + "_raw16.tif", bayer_array.demosaic()*64)
    cv2.imwrite(root_fname + "_raw8.png", (bayer_array.demosaic()//4).astype(np.uint8))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <filename.jpg> ...".format(sys.argv[0]))
        print("Specify one or more filenames corresponding to Raspberry Pi JPEGs including raw Bayer data.")
        print("Each file will be processed, to produce three new files:")
        print("<filename>_raw16.tif will contain the full raw data as a 16-bit TIFF file (the lower 6 bits are empty).")
        print("<filename>_raw8.png will contain the top 8 bits of the raw data, in an easier-to-handle file.")
        print("<filename>_exif.txt will contain the EXIF metadata extracted as a text file - this includes analogue gain.")
        sys.exit(0)
        
    for filename in sys.argv[1:]:
        extract_file(filename)