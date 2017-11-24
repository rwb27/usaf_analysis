from __future__ import print_function
import PIL.Image
import PIL.ExifTags
import sys
import os

def formatted_exif_data(image):
    """Retrieve an image's EXIF data and return as a dictionary with string keys"""
    # with thanks to https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
    exif_data = {}
    for k, v in image._getexif().items():
        # Use names rather than codes where they are available
        try:
            exif_data[PIL.ExifTags.TAGS[k]] = v
        except KeyError:
            exif_data[k] = v
    return exif_data
    
def parse_maker_note(maker_note):
    """Split the "maker note" EXIF field from a Raspberry Pi camera image into useful parameters"""
    camera_parameters = {}
    last_key = None
    for p in maker_note.split(" "):
        # The maker note contains <thing>=<thing> entries, space delimited but with spaces in some values.
        # So, if there is an = then we assume we've got key=value, but if there is no = sign, we append
        # the current chunk to the latest value, because that's where it probably belongs...
        if "=" in p:
            last_key, v = p.split("=")
            camera_parameters[last_key] = v
        else:
            camera_parameters[last_key] += " " + p
    return camera_parameters
    
def print_kv(k, v, format=""):
    """Consistently print a key-value pair"""
    print(("{0: >28}: {1" + format + "}").format(k, v))

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 2
        assert os.path.isfile(sys.argv[1])
    except:
        print("Usage: {} <filename.jpg>".format(sys.argv[0]))
        print("This tool will print out a summary of the image's metadata.  It's designed to work with images taken on a Raspberry Pi camera.")
        exit(-1)
        
    image = PIL.Image.open(sys.argv[1])
    exif_data = formatted_exif_data(image)
    for k, v in exif_data.iteritems():
        print_kv(k, v)
        
    camera_parameters = parse_maker_note(exif_data['MakerNote'])
    print("MakerNote Expanded:")
    for k, v in camera_parameters.iteritems():
        print_kv(k, v)
        
    print("Derived Values:")
    useful_metadata = {}
    # calculate exposure time - see https://www.media.mit.edu/pia/Research/deepview/exif.html
    ssv = exif_data['ShutterSpeedValue']
    exposure_time = 1/2.0**(float(ssv[0])/ssv[1])
    useful_metadata['exposure_time'] = exposure_time
    print_kv("exposure_time", exposure_time, ":.4")