#!/usr/bin/env python
#
# Activity Learning
#
# Copyright (C) 2015 Tinghui Wang <tinghui.wang@wsu.edu>
# 
# License: 3-Clause BSD
#
# Package Setup and Compilation

PACKAGE_NAME = 'actlearn'
PACKAGE_DISCRIPTION = 'Activity Learning Package'
MAINTAINER = 'Tinghui Wang (Steve)'
MAINTAINER_EMAIL = 'tinghui.wang@wsu.edu'
PACKAGE_URL = 'http://casas.wsu.edu/'
PACKAGE_LIC = '3-Clause BSD'
PACKAGE_AUTHOR = 'Tinghui Wang, et al.'
PACKAGE_DOWNLOAD_URL = 'to be announced'
PACKAGE_PLATFORMS = ['Windows', 'Linux', 'Mac OS X']
PACKAGE_CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python :: 2.7
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Information Analysis
Topic :: Software Development
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
"""
with open('README.rst', 'r') as f:
    PACKAGE_LONG_DISCRIPTION = f.read()


def is_numpy_installed():
    """
    Test and see if NumPy Is Installed
    """
    try:
        import numpy
    except ImportError:
        return False
    return True


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('actlearn')
    return config


def setup_package():
    metadata = dict(
        name=PACKAGE_NAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=PACKAGE_DISCRIPTION,
        long_description=PACKAGE_LONG_DISCRIPTION,
        url=PACKAGE_URL,
        author=PACKAGE_AUTHOR,
        download_url=PACKAGE_DOWNLOAD_URL,
        license=PACKAGE_LIC,
        classifiers=[_f for _f in PACKAGE_CLASSIFIERS.split('\n') if _f],
        platforms=PACKAGE_PLATFORMS,
        install_requires=[
          'numpy', 'theano', 'datetime'
        ],
    )

    if is_numpy_installed() is False:
        raise ImportError('Numerical Python (NumPy) is not installed.')

    metadata['configuration'] = configuration

    from numpy.distutils.core import setup

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
