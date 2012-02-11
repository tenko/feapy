feapy
=====

Simple linear static analysis of 3D frames and trusses.

feapy reads input files in a subset of the
Calculix/Abaqus format and create results
files in the frd format readably by Calculix's
cgx application for results viewing.

Currently Calculix expands beam elements to volume
elements and therfore can be unpractical for larger
models and complicated beam sections. Large beam models
are typical for ship structures, offshore structures
and the building industry. The long term goal of feapy
is to fill a small part of this gap and perhaps support
shell elements, non-linearities etc. in the future.

Initially feapy needs a python 2.7 installation including a
numpy installation. Other python version have not been tested.

Also for equation solving either:

 - Shipped solver wrapping the spooles solver.
 - pysparse package compiled with the superlu solver.
 - scipy package (A bug in scipy prevent this currently from working)

Later more of the numerical part of the code will be moved to faster
compiled Cython based code and the dependency on numpy will be removed.