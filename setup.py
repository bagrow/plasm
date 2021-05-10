#!/usr/bin/env python
# -*- coding: utf-8 -*-

# setup.py
# Jim Bagrow
# Last Modified: 2021-05-10

import codecs
import os.path
import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PLASM",
    version=get_version("plasm/__init__.py"),
    author="Jim Bagrow",
    author_email="bagrowjp@gmail.com",
    description="PLot Analysis Spreads for Meetings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bagrow/plasm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
    ],
)
