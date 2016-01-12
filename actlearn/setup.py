#!/usr/bin/env python
#
# Activity Learning
#
# Copyright (C) 2015 Tinghui Wang <tinghui.wang@wsu.edu>
#
# License: 3-Clause BSD
#
# Package Setup and Compilation

from __future__ import division, print_function

PACKAGE_NAME = 'actlearn'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(PACKAGE_NAME, parent_package, top_path)
    config.add_subpackage('utils')
    config.add_subpackage('data')
    config.add_subpackage('feature')
    config.add_subpackage('models')
    config.add_subpackage('training_algorithms')
    config.add_subpackage('log')
    config.add_subpackage('decision_tree')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
