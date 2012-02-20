# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
import sys
import unittest

try:
    from base import DofSet
except ImportError:
    sys.path.insert(0,'..')

from base import Dx, Dy, Dz, Rx, Ry, Rz
from base import Fx, Fy, Fz, Mx, My, Mz
from base import DofSet, BoundCon

class Test(unittest.TestCase):
    def almostEqual(self, a, b, places = 7):
        for va,vb in zip(a,b):
            self.assertAlmostEqual(va, vb, places)
            
class test_DofSet(Test):
    def test_dofset_reset(self):
        dof = DofSet()
        self.assertTrue(dof.count() == 0)
        dof.reset(True)
        self.assertTrue(dof.count() == 6)
        dof.reset()
        self.assertTrue(dof.count() == 0)
        
    def test_dofset_3d(self):
        dof1 = DofSet(Dx=True,Dy=True,Dz=True)
        self.assertTrue(dof1.count() == 3)
        self.assertTrue(dof1.is3DSolid())
        self.assertTrue(not dof1.is3DShell())
        self.assertTrue(dof1.elementDimension() == 3)
        
        dof2 = DofSet(Dx=True,Dy=True,Dz=True,Rx=True,Ry=True,Rz=True)
        self.assertTrue(dof2.count() == 6)
        self.assertTrue(not dof2.is3DSolid())
        self.assertTrue(dof2.is3DShell())
        self.assertTrue(dof2.elementDimension() == 3)
        
        dofa = dof1 & dof2
        self.assertTrue(dofa.count() == 3)
        self.assertTrue(dofa.is3DSolid())
        self.assertTrue(not dofa.is3DShell())
        self.assertTrue(dofa.elementDimension() == 3)
        
        dofo = dof1 | dof2
        self.assertTrue(dofo.count() == 6)
        self.assertTrue(not dofo.is3DSolid())
        self.assertTrue(dofo.is3DShell())
        self.assertTrue(dofo.elementDimension() == 3)
    
    def test_dofset_inplace(self):
        dof1 = DofSet(Dx=True,Dy=True,Dz=True)
        dof2 = DofSet(Dx=True,Dy=True,Dz=True,Rx=True,Ry=True,Rz=True)
        dof1 &= dof2
        self.assertTrue(dof1.count() == 3)
        self.assertTrue(dof1.is3DSolid())
        self.assertTrue(not dof1.is3DShell())
        self.assertTrue(dof1.elementDimension() == 3)
        
        dof1 = DofSet(Dx=True,Dy=True,Dz=True)
        dof2 = DofSet(Dx=True,Dy=True,Dz=True,Rx=True,Ry=True,Rz=True)
        dof1 |= dof2
        self.assertTrue(dof1.count() == 6)
        self.assertTrue(not dof1.is3DSolid())
        self.assertTrue(dof1.is3DShell())
        self.assertTrue(dof1.elementDimension() == 3)
    
    def test_dofset_2d(self):
        dof1 = DofSet(Dx=True,Dy=True,Dz=False)
        self.assertTrue(dof1.count() == 2)
        self.assertTrue(not dof1.is3DSolid())
        self.assertTrue(not dof1.is3DShell())
        self.assertTrue(dof1.elementDimension() == 2)
        
        dof2 = DofSet(Dx=True,Dy=True,Dz=True,Rx=True,Ry=True,Rz=True)
        
        dofa = dof1 & dof2
        self.assertTrue(dofa.count() == 2)
        self.assertTrue(not dofa.is3DSolid())
        self.assertTrue(not dofa.is3DShell())
        self.assertTrue(dofa.elementDimension() == 2)
        
        dofo = dof1 | dof2
        self.assertTrue(dofo.count() == 6)
        self.assertTrue(not dofo.is3DSolid())
        self.assertTrue(dofo.is3DShell())
        self.assertTrue(dofo.elementDimension() == 3)
    
    def test_dofset_dofPos(self):
        dof = DofSet(Dx=True,Dy=True,Dz=True)
        self.assertTrue(dof.dofPos(3) == 2)
        
        dof = DofSet(Dx=False,Dy=False,Dz=False,Rx=True,Ry=True,Rz=True)
        self.assertTrue(dof.dofPos(3) == 5)

class test_BoundCon(Test):
    def test_boundcon_dof(self):
        fixed = BoundCon(Dx=0.,Dy=0.,Dz=0.,Rx=0.,Ry=0.,Rz=0.)
        self.assertTrue(fixed[Dx] == 0.)
        self.assertTrue(fixed[Dz] == 0.)
        self.assertTrue(fixed[Dy] == 0.)
        self.assertTrue(fixed[Rx] == 0.)
        self.assertTrue(fixed[Ry] == 0.)
        self.assertTrue(fixed[Rz] == 0.)
        
        active = fixed.activeDofSet()
        self.assertTrue(active.count() == 0)
        
        del fixed[Dz]
        active = fixed.activeDofSet()
        self.assertTrue(active.count() == 1)
        
        fixed[Dz] = 0.
        active = fixed.activeDofSet()
        self.assertTrue(active.count() == 0)
        
        fixed[Dz] = 0.01
        active = fixed.activeDofSet()
        self.assertTrue(active.count() == 1)
        
        pinned = BoundCon(Dx=0.,Dy=0.,Dz=0.)
        self.assertTrue(pinned[Dx] == 0.)
        self.assertTrue(pinned[Dz] == 0.)
        self.assertTrue(pinned[Dy] == 0.)
        
        active = pinned.activeDofSet()
        self.assertTrue(active.count() == 3)
        
        del pinned[Dz]
        active = pinned.activeDofSet()
        self.assertTrue(active.count() == 4)
        
        pinned[Dz] = 0.
        active = pinned.activeDofSet()
        self.assertTrue(active.count() == 3)
        
        pinned[Dz] = 0.01
        active = pinned.activeDofSet()
        self.assertTrue(active.count() == 4)
    
    def test_boundcon_loads(self):
        load = BoundCon(Dx=0.01, Fz = 1000.)
        self.assertTrue(load[Dx] == 0.01)
        self.assertTrue(load[Fz] == 1000.)
        
        active = load.activeLoads([None,]*12)
        self.almostEqual(active,(0.01,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,1000.,
                                  0.,0.,0.))
        del load[Dx]
        active = load.activeLoads([None,]*12)
        self.almostEqual(active,(0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,1000.,
                                  0.,0.,0.))
        load[Dx] = 0.001
        active = load.activeLoads([None,]*12)
        self.almostEqual(active,(0.001,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,1000.,
                                  0.,0.,0.))
                                  
if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()