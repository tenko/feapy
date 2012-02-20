# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#-
import sys
import unittest
from math import sqrt

import numpy as np

try:
    from base import Node
except ImportError:
    sys.path.insert(0,'..')

from base import FE, Node, BoundCon, Properties, Material
from beam import Beam

class test_Beam(unittest.TestCase):
    def almostEqual(self, a, b, places = 7):
        for va,vb in zip(a,b):
            self.assertAlmostEqual(va, vb, places)
    
    def test_beam_init(self):
        n1 = Node(0.,0.,0.)
        n2 = Node(10.,10.,0.)

        b1 = Beam(n1,n2)
        b1.properties['Area'] = 0.1*0.1
        
        self.assertAlmostEqual(b1.length(), sqrt(200.))
        self.assertAlmostEqual(b1.volume(), 0.01*sqrt(200.))
        self.assertTrue(b1.sizeOfEM() == 12)
    
    def test_beam_transform(self):
        n1 = Node(0.,0.,0.)
        n2 = Node(10.,10.,0.)

        b1 = Beam(n1,n2)
        
        T = b1.calcT()
        
        p0 = np.dot(T,(0.,0.,0.))
        self.almostEqual(p0, (0.,0.,0.))
        
        p1 = np.dot(T,(10.,10.,0.))
        self.almostEqual(p1, (sqrt(200.),0.,0.))
        
if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()