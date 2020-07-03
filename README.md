# dvgutils
> Go Deep with your Vision

A set of tools to help you easily build your image and video processing pipeline using OpenCV and Python3.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

I suggest to use Python virtual environment. Virtual environments will allow you to run independent Python environments in isolation on your system.
Please take a look at [this article on RealPython](https://realpython.com/python-virtual-environments-a-primer/).

### Installing

Install [requirements.txt](requirements.txt) in your Python environment.

    $ pip install -r requirements.txt
    
### Development setup

    $ pip install -e .

## Running tests

    $ make test
    
## Running command line utils

Convert a video file to set of images:

    $ dvg-utils v2i -i assets/videos/faces.small.mp4 -o output --display
    
Convert set of frame images to a video file:

    $ dvg-utils i2v -i output -o output/my_new_file.avi --display
 
## Running examples

    $ python ./examples/capture_image.py -i assets/images/friends
    
    $ python ./examples/capture_video.py
    
    $ python ./examples/detect_face_image.py -i assets/images/friends --pipeline
    
    $ python ./examples/detect_face_video.py --metrics output/metrics.csv
    
    $ python ./examples/detect_face_video.py --pipeline --metrics output/metrics_pipeline.csv
    
    $ dvg-utils pm --input output/metrics.csv --input output/metrics_pipeline.csv 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [PyImageSearch](https://www.pyimagesearch.com/)
