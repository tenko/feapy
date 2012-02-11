# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from cython cimport array

cdef extern from "spooles.h":
    cdef struct factorinfo:
        pass
    
    void *spooles_factor(int *row, int *col, double *data,
                         int neq, int nnz, int symmetryflag)
    void spooles_solve(void *ptr, double *b, int neq)
    void spooles_cleanup(void *ptr)

cdef class Spooles:
    cdef void *ptr
    cdef readonly int neq
    
    def __init__(self, mat, int symflag = 0):
        coo = mat.toCOO()
        assert coo.shape[0] == coo.shape[1], 'Expected square matrix'
        self.neq = coo.shape[0]
        
        cdef double[:] data_view = coo.data
        cdef int[:] row_view = coo.row
        cdef int[:] col_view = coo.col
        
        self.ptr = spooles_factor(&row_view[0], &col_view[0], &data_view[0],
                                  coo.shape[0], coo.nnz, symflag)
    
    def __dealloc__(self):
        spooles_cleanup(self.ptr)
    
    def solve(self, b):
        cdef double[:] data = b
        spooles_solve(self.ptr, &data[0], self.neq)