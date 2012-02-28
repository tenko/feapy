# -*- coding: utf-8 -*-
#
# SPMatrix - sparse matrix classes
#
# cython global directives
# cython: wraparound=False
# cython: embedsignature=True
#
cimport cython
from cython.view cimport array

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memset
    
SIZEOF = {
    'b' : sizeof(byte),
    'B' : sizeof(ubyte),
    'i' : sizeof(int),
    'l' : sizeof(long),
    'f' : sizeof(float),
    'd' : sizeof(double),
    'z' : sizeof(zcomplex),
    'Z' : sizeof(Zcomplex),
}

class SPMatrixError(Exception):
    pass
    
cdef class SPMatrix:
    '''
    Sparse matrix base class
    '''
    def __init__(self, shape, format = None, bint isSym = False):
        self.shape = shape
        self.format = format
        self.isSym = isSym
        
        if shape[0] <= 0:
            raise SPMatrixError('rows <= 0')
        
        if shape[1] <= 0:
            raise SPMatrixError('cols <= 0')
        
        if format not in SIZEOF:
            raise SPMatrixError("unknown format '%s'" % format)
        
        if isSym and shape[0] != shape[1]:
            raise SPMatrixError('Expected square matrix')
        
    def __repr__(self):
        args = self.shape, self.nnz, self.format, self.isSym
        return "<shape=%s, nnz=%d, format=%s, isSym=%s>" % args
    
    property dtype:
        def __get__(self):
            return self.format
    
    property ndim:
        def __get__(self):
            return 2
    
    cpdef getRowCol(self, tuple key):
        '''Check row,col tuple argument'''
        cdef long row, col
        cdef long rows, cols
        
        rows, cols = self.shape
        
        try:
            row, col = key
        except (ValueError, TypeError):
            raise TypeError('index must be a pair of integers')
        
        if row < 0:
            row += rows
        
        if col < 0:
            col += cols
            
        if row < 0 or row >= rows:
            raise IndexError('index out of bounds')
        
        if col < 0 or col >= cols:
            raise IndexError('index out of bounds')
        
        if self.isSym and row < col:
            raise SPMatrixError('Access to upper part of symetric matrix')
        
        return row, col
        
include "ll.pxi"
include "coo.pxi"
include "csr.pxi"
include "cnv.pxi"
include "matops.pxi"