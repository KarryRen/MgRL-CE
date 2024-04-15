# -*- coding: utf-8 -*-
# @author : RenKai
# @time   : 2023/11/10 10:01
#
# pylint: disable=no-member

""" Compile the cython file. """

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

# ---- The price-alignment-features ---- #
setup(
    name="cal_paf",
    ext_modules=cythonize("cal_paf.pyx", language_level=3),
    include_dirs=[np.get_include()]
)
