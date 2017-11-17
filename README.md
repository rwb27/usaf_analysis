# USAF Analysis Scripts

Something I do quite a lot of, as part of the [Openflexure Microscope](https://www.github.com/rwb27/openflexure_microscope) project, is characterise microscopes.  Currently, these scripts are how I do it.

## Usage
To analyse a picture of a USAF image, and output the pixel size of the camera in nanometres:
```bash
python analyse_usaf_image.py <filename>
```

To analyse edge images for distortion, you need two folders, ``distortion_h`` and ``distortion_v`` inside your data folder, and then run:
```bash
python analyse_distortion.py <data_folder>
```

To analyse an edge image for resolution, just run:
```bash
python analyse_edge_image.py <filename>
```
You can also specify a folder, and it will create a PDF that analyses all the files in that folder.

The scripts can generally be used to analyse either a single measurement (by passing a folder of images, or a single image) or multiple measurements (by providing multiple folders/files, or one folder that contains many such folders/files).  All the scripts should display usage information when run with no arguments.

## Background
Generally I want to measure:
* The size of pixels, in sample units (i.e. nm per pixel)
* The field of view (in units of distance)
* The resolution of the microscope (using the point spread function)
* Any distortion that might be present
* Field curvature

The US Air Force resolution test target consists of a series of "elements" of three horizontal or vertical bars.  Usually the target is used by determining the smallest resolvable element, noting its number (given as a "group" and within that an "element" number), and finding the pitch of the three lines.  However, this doesn't work so well for microscopes, as the smallest element on most easily available targets is usually group 7, element 6, which has a pitch of 4.4um, i.e. the bars and spaces in between are 2.2um.  This is resolvable in even a fairly modest microscope.

### Pixel size/FoV
Instead, this script uses the USAF target as a convenient object of known size - it will automatically locate the elements in an image, and accurately determine their size in pixels.  From this, so long as the smallest element is known (the code assumes it's the group 7 element 6, though that can be changed), we can figure out the pixel size and field of view.

### Resolution
To determine resolution, we take images of an edge - it's convenient to use one of the large squares provided on the USAF target for this.  After acquiring images ad different values of defocus and different positions of the edge, we can differentiate the images to get the response to a point source (assuming the edge was sharp) and recover the point spread function.

### Distortion
Distortion can also be measured using an edge; assuming the distortion is primarily radial, a straight edge will appear to bend as it is translated across the field of view.  The distortion analysis script takes folders of images where an edge is scanned across the field of view, and recovers the distortion of the line as it is moved.


All these scripts are (c) Richard Bowman 2017 and shared under the GPL v3.0

