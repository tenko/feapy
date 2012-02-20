# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from cython cimport array

cdef class DOK(dict):
    cdef readonly tuple shape

    cdef matvec_(self, double *x, double *y, bint isSym = ?)
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y, bint isSym = ?)
    
    cpdef DOK toDOK(self, bint copy = ?)
    cpdef COO toCOO(self, bint copy = ?)
    cpdef CSR toCSR(self, bint copy = ?)

cdef class COO:
    cdef public array data
    cdef public array row
    cdef public array col
    cdef readonly int nnz
    cdef readonly tuple shape
    
    cdef matvec_(self, double *x, double *y)
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y)
    
    cpdef DOK toDOK(self, bint copy = ?)
    cpdef COO toCOO(self, bint copy = ?)
    cpdef CSR toCSR(self, bint copy = ?)

cdef class CSR:
    cdef public array data
    cdef public array indptr
    cdef public array indices
    cdef readonly int nnz
    cdef readonly tuple shape
    
    cdef matvec_(self, double *x, double *y)
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y)
    
    cpdef DOK toDOK(self, bint copy = ?)
    cpdef COO toCOO(self, bint copy = ?)
    cpdef CSR toCSR(self, bint copy = ?)