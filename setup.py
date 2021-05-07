#!/usr/bin/env python
# -*- coding: utf-8 -*-

# setup.py
# Jim Bagrow
# Last Modified: 2021-05-07

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PLASM",
    version="1.0.dev0",
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
