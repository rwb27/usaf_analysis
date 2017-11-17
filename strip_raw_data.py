"""
This is a brutally simple hack to chop the raw data off a Raspberry Pi JPEG file.

It assumes you are using v2 of the camera, and a full-resolution (i.e. no hardware binning) mode.

(c) 2017 Richard Bowman, released into the public domain with no conditions.
"""
from __future__ import print_function
import os
import sys

if __name__ == "__main__":
    try:
        assert len(sys.argv) >= 2
        for fname in sys.argv[1:]:
            assert os.path.isfile(fname)
    except:
        print("Usage: {} <image_filename> [<image_filename> ...]".format(sys.argv[0]))
        print("Images should be jpeg files, acquired at full resolution on a Pi Camera v2.")
        print("For each input image, a smaller file will be generated as <filename>_noraw.jpeg")
        
    for fname in sys.argv[1:]:
        bits = fname.rsplit('.',1)
        filesize = os.stat(fname).st_size
        oname = ".".join([bits[0]+"_noraw"] + bits[1:]) #output filename has _noraw before .jpeg
        with open(fname, "rb") as input, open(oname, "wb") as output:
            output.write(input.read(filesize - 10270208))
            