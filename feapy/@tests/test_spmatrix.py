# -*- coding: utf-8 -*-
# This file is part of feapy - See LICENSE.txt
#-
import sys
import unittest

import numpy as np

try:
    import spmatrix
except ImportError:
    sys.path.insert(0,'..')

import spmatrix

class test_spmatrix(unittest.TestCase):
    def almostEqual(self, a, b, places = 7):
        for va,vb in zip(a,b):
            self.assertAlmostEqual(va, vb, places)
            
    def setUp(self):
        dok = self.dok = spmatrix.DOK((5,5))
        for col in range(5):
            for row in range(3, col, -1):
                dok[row,col] = 1.

        for col in range(5):
            for row in range(2, col, -1):
                dok[row,col] += 1.5

        for row in range(5):
            dok[row,row] = row + 5
    
    def test_matvec(self):
        vec = np.arange(1, 6, dtype=float)
        exp = (5., 14.5, 28.5, 38., 45.)
        ret = np.zeros((5,), dtype=float)
        
        self.dok.matvec(vec, ret)
        self.almostEqual(ret, exp)
        
        coo = self.dok.toCOO()
        vec = np.arange(1, 6, dtype=float)
        ret[:] = 0.
        coo.matvec(vec, ret)
        self.almostEqual(ret, exp)
        
        cpy = coo.toCOO(copy=True)
        vec = np.arange(1, 6, dtype=float)
        ret[:] = 0.
        cpy.matvec(vec, ret)
        self.almostEqual(ret, exp)
        
        csr = cpy.toCSR()
        vec = np.arange(1, 6, dtype=float)
        ret[:] = 0.
        csr.matvec(vec, ret)
        self.almostEqual(ret, exp)
        
        cpy = csr.toCOO(copy=True)
        vec = np.arange(1, 6, dtype=float)
        ret[:] = 0.
        cpy.matvec(vec, ret)
        self.almostEqual(ret, exp)
        
        cpy = csr.toCOO(copy=False)
        vec = np.arange(1, 6, dtype=float)
        ret[:] = 0.
        cpy.matvec(vec, ret)
        self.almostEqual(ret, exp)
    
    def test_matvec_sym(self):
        vec = np.arange(1, 6, dtype=float)
        exp = (21.5, 26., 32.5, 38., 45.)
        ret = np.zeros((5,), dtype=float)
        
        self.dok.matvec(vec, ret, True)
        self.almostEqual(ret, exp)
        
        coo = self.dok.toCOO()
        vec = np.arange(1, 6, dtype=float)
        ret[:] = 0.
        coo.matvec(vec, ret, True)
        self.almostEqual(ret, exp)
        
if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()