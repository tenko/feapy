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
from Cython.Distutils import build_ext

sys.argv.append('build_ext')
sys.argv.append('--inplace')

'''
MKLROOT = os.environ["MKLROOT"]
MKL_LINK = [
    "-fopenmp",
    "-Wl,--start-group",
    os.path.join(MKLROOT, 'lib', 'intel64', 'libmkl_intel_lp64.a'),
    os.path.join(MKLROOT, 'lib', 'intel64', 'libmkl_gnu_thread.a'),
    os.path.join(MKLROOT, 'lib', 'intel64', 'libmkl_core.a'),
    "-Wl,--end-group",
]
'''

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
            #extra_objects=["@src/libarpack.a"],
            sources=["@src/solver.c", "@src/spooles.c"],
            include_dirs = ["@src", r'/usr/include/spooles',],
            libraries=["spoolesMT", "spooles", "lapack", "blas", "pthread"],
            #libraries=["spoolesMT", "spooles", "pthread", "gfortran"],
            extra_compile_args = ["-DUSE_MT"],
            #extra_link_args = MKL_LINK,
        ),
        ],
      cmdclass = {'build_ext': build_ext}
    )
except:
    print('Traceback\n:%s\n' % str(sys.exc_info()[-2]))
else:
    print('\n')