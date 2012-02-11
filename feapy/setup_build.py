# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
# Note : Currently a trunk version of Cython is needed
#        Therfore the resuting .c code is shipped.
#
import sys
import os

from distutils.core import setup
from distutils.extension import Extension
#from Cython.Distutils import build_ext

sys.argv.append('build_ext')
sys.argv.append('--inplace')

try:
    setup(
      ext_modules=[
        Extension(
            "spmatrix",
            #sources=["@src/spmatrix.pyx"],
            sources=["@src/spmatrix.c"],
            include_dirs = ["@src"],
        ),
        Extension(
            "solver",
            #sources=["@src/solver.pyx", "@src/spooles.c"],
            sources=["@src/solver.c", "@src/spooles.c"],
            include_dirs = ["@src", r'/usr/include/spooles',],
            libraries=["spoolesMT", "spooles", "pthread"],
            extra_compile_args = ["-DUSE_MT"],
        ),
        ],
      #cmdclass = {'build_ext': build_ext}
    )
except:
    print('Traceback\n:%s\n' % str(sys.exc_info()[-2]))
else:
    print('\n')