# dvgutils
> Go Deep with your Vision

**dvgutils** is a set of tools to help you easily build your image and video processing pipeline using OpenCV and Python3.
With an easy configuration file and implementation modules you can try your computer vision pipeline 
on different devices like webcams, RaspberryPi Camera or camera connected to 
NVIDIA Jetson device (Nano, TX2, XAVIER) through GStreamer and more.

The library provides a handful of examples to start from.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

### Prerequisites

I suggest to use Python virtual environment. Virtual environments will allow you to run independent Python environments 
in isolation on your system.  
Please take a look at [this article on RealPython](https://realpython.com/python-virtual-environments-a-primer/).

    $ pip install tqdm pyyaml numpy

Istall OpenCV with:

    $ pip install opencv-contrib
    
or

    $ pip install opencv-contrib-python
    
if you need both main and contrib modules.
    
Note: Skip this on Jetson devices with JetPack SDK as it is already installed.

### Installation

    $ pip install dvg-utils
    
### Development setup

We can install the package locally (for use on our system), with:

    $ pip install .
    
We can also install the package with a symlink, so that changes to the source files will be
immediately available to other users of the package on our system:

    $ pip install -e .   

## Running tests

Running tests requires `pytest` package to be installed.

    $ pip install pytest

    $ make test
    
## Command line utils

After `dvg-utils` installation you get access to `dvg-utils` command which allows you to:

- convert a video file to set of images:

      $ dvg-utils v2i -i assets/videos/faces.mp4 -o output --display
    
- convert set of frame images to a video file:

      $ dvg-utils i2v -i output -o output/my_new_file.avi --display
      
- plot metrics (see examples below):

      $ dvg-utils pm --input output/metrics_1.csv --input output/metrics_2.csv
 
## Examples

### Preparation

    $ pip install matplotlib
    $ pip install scipy
    $ pip install dlib

If you run examples on Raspberry Pi Camera you will need to:

    $ pip install picamera

### Capture single image

    $ python ./examples/capture_image.py -i assets/images/friends

### Capture video stream 

    $ python ./examples/capture_video.py
    
### Capture two video stream from web cameras

    $ python ./examples/capture_multi_video.py
    
### Detect faces in images

    $ python ./examples/detect_face_image.py -i assets/images/friends

### Detect faces in a video stream:
    
    $ python ./examples/detect_face_video.py --metrics output/metrics_no_pipeline.csv

or using pipeline implementation:
    
    $ python ./examples/detect_face_video.py --pipeline --metrics output/metrics_with_pipeline.csv

### Compare the collected metrics
    
    $ dvg-utils pm --input output/metrics_no_pipeline.csv --input output/metrics_with_pipeline.csv 
    
### Detect motion in a video stream
    
    $ python ./examples/detect_motion_video.py

### Detect object in a video stream

    $ python ./examples/detect_object_video.py

### Track object in a video stream

    $ python ./examples/track_object_video.py
    
### Count object in a video stream

    $ python ./examples/count_object_video.py

## Acknowledgments

* [PyImageSearch](https://www.pyimagesearch.com/) for ideas
* [pixabay](https://pixabay.com/) for assets

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
