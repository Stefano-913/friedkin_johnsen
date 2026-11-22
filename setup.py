#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup

__author__ = ['Stefano Liazza']
__email__ = ['stefano.liazza@studio.unibo.it']

PACKAGE_NAME = 'friedkin-johnsen'
MODULE_NAME = 'friedkin_johnsen'
PACKAGE_VERSION = '0.0.1'
DESCRIPTION = 'Simulation of Friedkin-Johnsen opinion model.'
AUTHOR = 'Stefano Liazza'
EMAIL = 'stefano.liazza@studio.unibo.it'
REQUIRES_PYTHON = '>=3'
URL = 'https://github.com/Stefano-913/friedkin_johnsen'
DOWNLOAD_URL = URL

setup(
  name=PACKAGE_NAME,
  version=PACKAGE_VERSION,
  description=DESCRIPTION,
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  install_requires=[],
  url=URL,
  download_url=DOWNLOAD_URL,
  setup_requires=[],
  packages=[
    MODULE_NAME,
    f"{MODULE_NAME}.opinion_models",
  ],
  package_data={
    MODULE_NAME: [],
  },
  include_package_data=True,
  platforms='any',
  classifiers=[
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Programming Language :: Python :: 3.14',
    'Programming Language :: Python :: Implementation :: CPython',
    'Programming Language :: Python :: Implementation :: PyPy'
  ],
  entry_points={'console_scripts': [
    'friedkin-johnsen = friedkin_johnsen.__main__:main',
    ],
  },
)