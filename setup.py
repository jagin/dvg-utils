from setuptools import setup, find_packages
import os
import re

root = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(root, "README.md"), "r") as readme_file:
    long_description = readme_file.read()

with open(os.path.join(root, "dvgutils", "__init__.py"), "r") as init_file:
    init_content = init_file.read()
attrs = dict(re.findall(r"__([a-z]+)__ *= *['\"](.+)['\"]", init_content))

setup(
    name="dvg-utils",
    version=attrs['version'],
    author="Jarosław Gilewski",
    author_email="jgilewski@jagin.pl",
    description="DeepVisionGuru utilities for image/video capturing and processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jagin/dvg-utils",
    packages=find_packages(include=["dvgutils", "dvgutils.*"]),
    keywords=["computer vision", "image processing", "opencv"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=["bin/dvg-utils"],
    python_requires='>=3.6'
)
