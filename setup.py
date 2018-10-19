# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setuptools installation script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

description = """Graph Nets is DeepMind's library for building graph networks in
Tensorflow and Sonnet.
"""

setup(
    name="graph_nets",
    version="1.0.1",
    description="Library for building graph networks in Tensorflow and Sonnet.",
    long_description=description,
    author="DeepMind",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "tensorflow", "sonnet", "machine learning"],
    url="https://github.com/deepmind/graph-nets",
    packages=find_packages(),
    install_requires=[
        "absl-py",
        "dm-sonnet",
        "future",
        "networkx",
        "numpy",
        "setuptools",
        "six",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
