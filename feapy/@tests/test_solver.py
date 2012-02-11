# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#-
import sys
import unittest
from math import sqrt

import numpy as np

try:
    import spmatrix
except ImportError:
    sys.path.insert(0,'..')

import spmatrix
import solver

class test_Solver(unittest.TestCase):
    def almostEqual(self, a, b, places = 7):
        for va,vb in zip(a,b):
            self.assertAlmostEqual(va, vb, places)
    
    def test_solver(self):
        dok = spmatrix.DOK()
        for col in range(5):
            for row in range(4, col, -1):
                dok[row,col] = 1.

        for col in range(5):
            for row in range(4, col, -1):
                dok[row,col] += 1.5

        for row in range(5):
            dok[row,row] = 5.

        rhs = np.array((1.,1.,1.,1.,1.), dtype=float)
        
        spooles = solver.Spooles(dok, 2)
        spooles.solve(rhs)
        
        self.almostEqual(rhs, [0.2, 0.1, 0.05, 0.025, 0.0125])

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()