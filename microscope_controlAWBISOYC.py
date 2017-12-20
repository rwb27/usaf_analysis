# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Simple stage control for Fergus Riche's "fergboard" controller.

Written 2016 by Richard Bowman, Abhishek Ambekar, James Sharkey and Darryl Foo

Released under GNU GPL v3 or later.

Usage:
    microscope move <x> <y> [<z>]
    microscope focus <z>
    microscope [options] control [<step_size>]
    microscope [options]

Options:
      --output=<filepath>   Set output directory/filename [default: ~/Desktop/images]
      -h --help             Show this screen.
"""

import io
import sys
import os
import time
import numpy as np
import docopt
import curses
import curses.ascii
import picamera
from openflexure_stage import OpenFlexureStage

def round_resolution(res):
    """Round up the camera resolution to units of 32 and 16 in x and y"""
    return tuple([int(q*np.ceil(res[i]/float(q))) for i, q in enumerate([32,16])])

def decimate_to(shape, image):
    """Decimate an image to reduce its size if it's too big."""
    decimation = np.max(np.ceil(np.array(image.shape, dtype=np.float)[:len(shape)]/np.array(shape)))
    return image[::decimation, ::decimation, ...]

def sharpness_sum_lap2(rgb_image):
    """Return an image sharpness metric: sum(laplacian(image)**")"""
    image_bw=np.mean(decimate_to((1000,1000), rgb_image),2)
    image_lap=ndimage.filters.laplace(image_bw)
    return np.mean(image_lap.astype(np.float)**4)

def sharpness_edge(image):
    """Return a sharpness metric optimised for vertical lines"""
    gray = np.mean(image.astype(float), 2)
    n = 20
    edge = np.array([[-1]*n + [1]*n])
    return np.sum([np.sum(ndimage.filters.convolve(gray,W)**2) 
                   for W in [edge, edge.T]])

class Microscope(object):
    def __init__(self, camera=None, stage=None):
        """Not bothering with context manager yet!"""
        self.cam = camera
        self.stage = stage

    def rgb_image_old(self, use_video_port=True):
        """Capture a frame from a camera and output to a numpy array"""
        res = round_resolution(self.cam.resolution)
        shape = (res[1], res[0], 3)
        buf = np.empty(np.product(shape), dtype=np.uint8)
        self.cam.capture(buf, 
                format='rgb', 
                use_video_port=use_video_port)
        #get an image, see picamera.readthedocs.org/en/latest/recipes2.html
        return buf.reshape(shape)

    def rgb_image(self, use_video_port=True, resize=None):
        """Capture a frame from a camera and output to a numpy array"""
        with picamera.array.PiRGBArray(self.cam, size=resize) as output:
            self.cam.capture(output, 
                    format='rgb', 
                    resize=resize,
                    use_video_port=use_video_port)
        #get an image, see picamera.readthedocs.org/en/latest/recipes2.html
            return output.array

    def freeze_camera_settings(self, iso=None, wait_before=2, wait_after=0.5):
        """Turn off as much auto stuff as possible"""
        if iso is not None:
            self.cam.iso = iso
        time.sleep(wait_before)
        self.cam.shutter_speed = self.cam.exposure_speed
        self.cam.exposure_mode = "off"
        g = self.cam.awb_gains
        self.cam.awb_mode = "off"
        #self.cam.awb_gains = g
        self.cam.awb_gains = [1.5,1.8]
        time.sleep(wait_after)

    def scan_linear(self, rel_positions, backlash=0, return_to_start=True):
        """Scan through a list of (relative) positions (generator fn)
        
        rel_positions should be an nx3-element array (or list of 3 element arrays).  
        Positions should be relative to the starting position - not a list of relative moves.

        
        backlash defines the final move used to approach each point.  If it is
        a scalar, we will expand to an xyz vector, but keep any compnents that 
        are not moving at zero.
        
        if return_to_start is True (default) we return to the starting position after a
        successful scan.  NB we always attempt to return to the starting position if the
        scan was unsuccessful.
        """
        starting_position = self.stage.position
        rel_positions = np.array(rel_positions)
        assert rel_positions.shape[1] == 3, ValueError("Positions should be 3 elements long.")
        try:
            assert backlash.shape == (3,), "Backlash should be a 3-element array"
        except:
            if backlash is None:
                backlash = 0
            backlash = int(backlash)
            # enable backlash correction for all axes that move.
            backlash = backlash * np.any(rel_positions != 0, axis=0).astype(int)
        try:
            self.stage.move_rel(rel_positions[0] - backlash)
            self.stage.move_rel(backlash)
            yield 0

            for i, step in enumerate(np.diff(rel_positions, axis=0)):
                self.stage.move_rel(step-backlash)
                self.stage.move_rel(backlash)
                yield i+1
        except Exception as e:
            return_to_start = True # always return to start if it went wrong.
            raise e
        finally:
            if return_to_start:
                self.stage.move_abs(starting_position)

    def scan_z(self, dz, backlash=0, return_to_start=True):
        """Scan through a list of (relative) z positions (generator fn)"""
        starting_position = self.stage.position
        try:
            self.stage.focus_rel(dz[0]-backlash)
            self.stage.focus_rel(backlash)
            yield 0

            for i, step in enumerate(np.diff(dz)):
                self.stage.focus_rel(step)
                yield i+1
        finally:
            if return_to_start:
                self.stage.move_abs(starting_position)


    def autofocus(self, dz, backlash=0, settle=0.5, metric_fn=sharpness_sum_lap2):
        """Perform a simple autofocus routine.

        The stage is moved to z positions (relative to current position) in dz,
        and at each position an image is captured and the sharpness function 
        evaulated.  We then move back to the position where the sharpness was
        highest.  No interpolation is performed.

        dz is assumed to be in ascending order (starting at -ve values)
        """
        sharpnesses = []
        positions = []
        def z():
            return self.stage.position[2]
        def measure():
            time.sleep(settle)
            sharpnesses.append(metric_fn(self.rgb_image(
                        use_video_port=True, 
                        resize=(640,480))))
            positions.append(z())

        self.stage.focus_rel(dz[0]-backlash)
        self.stage.focus_rel(backlash)
        measure()

        for step in np.diff(dz):
            self.stage.focus_rel(step)
            measure()
           
        newposition = positions[np.argmax(sharpnesses)]

        self.stage.focus_rel(newposition - backlash - z())
        self.stage.focus_rel(backlash)

        return positions, sharpnesses


def validate_filepath(filepath):
    """Check the filepath is valid, creating dirs if needed
    The final format is  ~/Desktop/images/image_%d.img  
    %d is formatted with number by (filepath %n)
    https://pyformat.info/"""

    filepath = os.path.expanduser(filepath)
    if "%d" not in filepath and ".jp" not in filepath:
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        return os.path.join(filepath, "image_%03d.jpg")

    elif "%d" not in filepath and ".jp" in filepath:
        'add automatic numbering to filename'
        filepath = filepath.split('.')
        filepath = filepath[0] + '_%03d.' + filepath[1] 
        return filepath
    
    elif "%d" in filepath and ".jp" in filepath:
        return filepath
    
    else:
        raise ValueError("Error setting output filepath.  Valid filepaths should"
                         " either be [creatable] directories, or end with a "
                         "filename that contains '%d' and ends in '.jpg' or '.jpeg'")






#run microscope_control.py directly
if __name__ == '__main__':    
    pass
    
    argv = docopt.docopt(__doc__, options_first=True)
    
    stage = OpenFlexureStage('/dev/ttyUSB0')

    if argv['move']:
        x, y, z = [int(argv.get(d, 0)) for d in ['<x>', '<y>', '<z>']]
        print ("moving", x, y, z)
        stage.move_rel([x, y, z])
    elif argv['focus']:
        stage.focus_rel(int(argv['<z>']))
    else: #if argv['control']:
        def move_stage_with_keyboard(stdscr):
            stdscr.addstr(0,0,"wasd to move in X/Y, qe for Z\n"
                          "r/f to decrease/increase step.\n"
                          "v/b to start/stop video preview.\n"
                          "i/o to zoom in/out.\n"
                          "t/g to adjust contrast, y/h to adjust brightness.\n"
                          "j to save jpeg file, k to change output path.\n"
                          "x to quit\n")
            step = int(argv.get('<step>',100))
            filepath = validate_filepath(argv['--output'])
            fov = 1
            with picamera.PiCamera(resolution=(3280/2,2464/2)) as camera:
                camera.start_preview()
                ms = Microscope(camera, stage)
                ms.freeze_camera_settings(iso=100)
                camera.shutter_speed = camera.shutter_speed / 4
                backlash=128
                #time.sleep(3)
                #camera.start_preview()
                #time.sleep(3)
                #camera.stop_preview()
                while True:
                    c = stdscr.getch()
                    if curses.ascii.isascii(c):
                        c = chr(c)
                    if c == 'x':
                        break
                    elif c == 'w' or c == curses.KEY_UP:
                        stage.move_rel([0,step,0])
                    elif c == 'a' or c == curses.KEY_LEFT:
                        stage.move_rel([step,0,0])
                    elif c == 's' or c == curses.KEY_DOWN:
                        stage.move_rel([0,-step,0])
                    elif c == 'd' or c == curses.KEY_RIGHT:
                        stage.move_rel([-step,0,0])
                    elif c == 'q' or c == curses.KEY_PPAGE:
                        stage.move_rel([0,0,-step])
                    elif c == 'e' or c == curses.KEY_NPAGE:
                        stage.move_rel([0,0,step])
                    elif c == "r" or c == '-' or c == '_':
                        step /= 2
                    elif c == "f" or c == '+' or c == '=':
                        step *= 2
                    elif c == 'i':
                        fov *= 0.75
                        camera.zoom = (0.5-fov/2, 0.5-fov/2, fov, fov)
                    elif c == 'o':
                        if fov < 1.0:
                            fov *= 4.0/3.0
                        camera.zoom = (0.5-fov/2, 0.5-fov/2, fov, fov)
                    elif c == 't':
                        if camera.contrast <= 90:
                            camera.contrast += 10
                    elif c == 'g':
                        if camera.contrast >= -90:
                            camera.contrast -= 10
                    elif c == 'y':
                        if camera.brightness <= 90:
                            camera.brightness += 10
                    elif c == 'h':
                        if camera.brightness >= -90:
                            camera.brightness -= 10
                    elif c == 'n':
                        if camera.shutter_speed <= 1000000:
                            camera.shutter_speed += 1000
                            camera.annotate_text= '{}'.format(camera.shutter_speed)
                    elif c == 'm':
                        if camera.shutter_speed >= 1000:
                            camera.shutter_speed -= 1000
                            camera.annotate_text= '{}'.format(camera.shutter_speed)
                    elif c == "v":
                        camera.start_preview()
                        camera.awb_mode="off"
                        camera.awb_gains=(1,1)
                        camera.annotate_text = "awb_gains: {}{}".format(camera.awb_gains[0],camera.awb_gains[1])
                    elif c == "b":
                        camera.stop_preview()
                    elif c == "j":
                        camera.annotate_text="Saving image..."
                        n = 0
                        while os.path.isfile(os.path.join(filepath % n)):
                            n += 1
                        camera.annotate_text="Saving '%s'" % (filepath % n)
                        camera.annotate_text="Saved '%s'" % (filepath % n)
                        camera.annotate_text="analog_gain: {}, digital_gain:{}, exposure_compensation: {}, ISO: {}, awb_gains: {}_{}, contrast: {}, brightness: {}, exposure_speed: {}, shutter_speed: {}".format(camera.analog_gain, camera.digital_gain, camera.exposure_compensation, camera.iso, camera.awb_gains[0], camera.awb_gains[1], camera.contrast, camera.brightness, camera.exposure_speed, camera.shutter_speed)
                        camera.capture(filepath % n, format="jpeg", bayer=True)
                        try:
                            stdscr.addstr("Saved '%s'\n" % (filepath % n))
                        except:
                            pass
                        time.sleep(0.5)
                        #camera.annotate_text=""
                    elif c == "p":
                        camera.annotate_text="Position '%s'" % str(stage.position)
                        try:
                            stdscr.addstr("Position '%s'\n" % str(stage.position))
                        except:
                            pass
                        time.sleep(0.5)
                        camera.annotate_text=""
                    elif c == "l":
                        camera.annotate_text="White balance {}".format(camera.awb_gains)
                        try:
                            stdscr.addstr("White balance {}".format(camera.awb_gains))
                        except:
                            pass
                        time.sleep(0.5)
                        camera.annotate_text=""
                    elif c == "k":
                        camera.stop_preview()
                        stdscr.addstr("The new output location can be a directory or \n"
                                      "a filepath.  Directories will be created if they \n"
                                      "don't exist, filenames must contain '%d' and '.jp'.\n"
                                      "New filepath: ")
                        curses.echo()
                        new_filepath = stdscr.getstr()
                        curses.noecho()
                        if len(new_filepath) > 3:
                            filepath = validate_filepath(new_filepath)
                        stdscr.addstr("New output filepath: %s\n" % filepath)

        curses.wrapper(move_stage_with_keyboard)

    


