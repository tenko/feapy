# -*- coding: utf-8 -*-
# This file is part of spmatrix - See LICENSE.txt
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
    
    def createMatrix(self, m):
        for col in range(5):
            for row in range(3, col, -1):
                m[row,col] = 1.

        for col in range(5):
            for row in range(2, col, -1):
                m[row,col] += 1.5

        for row in range(5):
            m[row,row] = row + 5
            
        return m
        
    def test_LL(self):
        dok = {}
        self.createMatrix(dok)
        
        M = spmatrix.LL((5,5), 5, format='d')
        self.createMatrix(M)
        
        # check
        for key, value in dok.iteritems():
            self.assertTrue(M[key] == value)
        
        # check copy constructs
        for typ in ('LL','COO','CSR'):
            for cpy in (True,False):
                m = getattr(M, 'to' + typ)(copy = cpy)
                mdok = m.toDOK()
                for key, value in dok.iteritems():
                    self.assertTrue(mdok[key] == value)
                
    def test_matvec(self):
        M = spmatrix.LL((5,5), 5, format='d')
        self.createMatrix(M)
        
        # check copy constructs
        for typ in ('LL','COO','CSR'):
            for cpy in (True,False):
                m = getattr(M, 'to' + typ)(copy = cpy)
                
                vec = np.arange(1, 6, dtype=float)
                ret = np.zeros((5,), dtype=float)
                m.matvec(vec, ret)
                self.almostEqual(ret, (5., 14.5, 28.5, 38., 45.))
        
        M = spmatrix.LL((5,5), 5, format='Z')
        self.createMatrix(M)
        
        # check copy constructs
        for typ in ('LL','COO','CSR'):
            for cpy in (True,False):
                m = getattr(M, 'to' + typ)(copy = cpy)
                
                vec = np.arange(1, 6, dtype=complex)
                ret = np.zeros((5,), dtype=complex)
                m.matvec(vec, ret)
                self.almostEqual(ret, (5.+0.j, 14.5+0.j, 28.5+0.j, 38.+0.j, 45.+0.j))
    
    def test_matvec_sym(self):
        M = spmatrix.LL((5,5), 5, isSym = True, format='d')
        self.createMatrix(M)
        
        # check copy constructs
        for typ in ('LL','COO','CSR'):
            for cpy in (True,False):
                m = getattr(M, 'to' + typ)(copy = cpy)
                
                vec = np.arange(1, 6, dtype=float)
                ret = np.zeros((5,), dtype=float)
                m.matvec(vec, ret)
                self.almostEqual(ret, (21.5, 26., 32.5, 38., 45.))
                
        M = spmatrix.LL((5,5), 5, isSym = True, format='Z')
        self.createMatrix(M)
        
        # check copy constructs
        for typ in ('LL','COO','CSR'):
            for cpy in (True,False):
                m = getattr(M, 'to' + typ)(copy = cpy)
                
                vec = np.arange(1, 6, dtype=complex)
                ret = np.zeros((5,), dtype=complex)
                m.matvec(vec, ret)
                self.almostEqual(ret, (21.5+0.j, 26.+0.j, 32.5+0.j, 38.+0.j, 45.+0.j))
    
if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main()