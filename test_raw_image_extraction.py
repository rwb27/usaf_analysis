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
        image = PIL.Image.open(filename)
        # with thanks to https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
        exif_data = img._getexif()
        for k, v in exif_data.iteritems():
            # duplicate items with nice names for keys and not just numeric codes, if they have nice names.
            if k in PIL.ExifTags.TAGS:
                exif_data[PIL.ExifTags.TAGS[k]] = v
        return bayer_array, jpeg, exif_data
    return bayer_array

if __name__ == "__main__":
    """jpeg = open("usaf_test.jpg", mode="rb").read()
    print("jpeg file was "+str(len(jpeg))+" bytes long")
    
    cam = DummyCam()
    
    bayer_array = picamera_array.PiBayerArray(cam)
    bayer_array.write(jpeg)
    bayer_array.flush()
    
    jpeg_array = cv2.imdecode(np.asarray(bytearray(jpeg), dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    """
    bayer_array, pil_image, exif = load_raw_image("test_raw.jpg", open_jpeg=True)
    jpeg_array = cv2.imread("test_raw.jpg", cv2.CV_LOAD_IMAGE_UNCHANGED)
    
    print(exif)
    
#    plt.imshow(np.sum(bayer_array.array, axis=2))
#    plt.figure()
#    plt.imshow(np.sum(jpeg_array, axis=2))

    bright_bayer = load_raw_image("bright_reference.jpg", picamera_array.PiFastBayerArray)
    bright = bright_bayer.demosaic()
    
    plt.imshow((bright).astype(np.float)/np.max(bright, axis=(0,1)).astype(np.float))
    plt.show()