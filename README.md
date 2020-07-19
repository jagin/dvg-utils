# dvgutils
> Go Deep with your Vision

A set of tools to help you easily build your image and video processing pipeline using OpenCV and Python3.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

I suggest to use Python virtual environment. Virtual environments will allow you to run independent Python environments in isolation on your system.
Please take a look at [this article on RealPython](https://realpython.com/python-virtual-environments-a-primer/).

Install [requirements.txt](requirements.txt) in your Python environment.

    $ pip install -r requirements.txt

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

    $ make test
    
## Command line utils

Convert a video file to set of images:

    $ dvg-utils v2i -i assets/videos/faces.mp4 -o output --display
    
Convert set of frame images to a video file:

    $ dvg-utils i2v -i output -o output/my_new_file.avi --display
 
## Examples

    $ python ./examples/capture_image.py -i assets/images/friends
    
    $ python ./examples/capture_video.py
    
    $ python ./examples/capture_multi_video.py

    $ python ./examples/detect_face_image.py -i assets/images/friends
    
    $ python ./examples/detect_face_video.py --metrics output/metrics_no_pipeline.csv
    
    $ python ./examples/detect_face_video.py --pipeline --metrics output/metrics_with_pipeline.csv
    
    $ dvg-utils pm --input output/metrics_no_pipeline.csv --input output/metrics_with_pipeline.csv 
    
    $ python ./examples/detect_motion_video.py

    $ python ./examples/detect_object_video.py

    $ python ./examples/track_object_video.py

    $ python ./examples/count_object_video.py

## Acknowledgments

* [PyImageSearch](https://www.pyimagesearch.com/) for ideas
* [pixabay](https://pixabay.com/) for assets

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
