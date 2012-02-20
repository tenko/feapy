Introduction
============

Simple linear static and modal analysis of 3D frames and trusses.

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

 - Shipped solver wrapping the spooles and arpack solver.
 - pysparse package compiled with the superlu solver.

Later more of the numerical part of the code will be moved to faster
compiled Cython based code and the dependency on numpy will be removed.

Elements
========

The following elements are supported:

 * B31 - Two node Timoshenko beam element which takes into account shear
   deformations if sheare areas are supplied.
 * T3D2 - Two node truss element. This element support only forces in the
   axial direction.
 * S3 - Non-structural triangle element used for calculating beam section
   properties. Please note that this calculation is at present very simple
   and does not calculate section shear properties or torsion correctly.
   Results should only be used for long slender beam elements.

Keywords
========

 * INCLUDE
 * NODE
 * NSET
 * ELEMENT Support only elements outlined above
 * ELSET
 * MATERIAL Support only ELASTIC and DENSITY material definitions.
 * BEAM SECTION Support only special general section with given section properties.
 * BOUNDARY
 * TRANSFORM
 * STEP
 * STATIC
 * FREQUENCY
 * CLOAD Only structural dof's 1-6
 * DLOAD Support GRAV, PX, PY, PZ, P2, P3. 
   PX, PY & PZ are line loads in global direction (E.g N/m) and
   P2/P3 are line loads in beam local axis directions.
 * NODE PRINT Support only U & RF results
 * NODE FILE Support only U & RF results
 * EL PRINT Support only SF and EVOL. SF are section forces along beam element
   with optional SECTIONDELTA distance along element given.
 * END STEP
