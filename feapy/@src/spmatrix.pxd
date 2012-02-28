# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from cython.view cimport array

ctypedef signed char byte
ctypedef unsigned char ubyte
ctypedef unsigned short ushort
ctypedef float complex zcomplex
ctypedef double complex Zcomplex

ctypedef fused numeric:
    float
    double
    zcomplex
    Zcomplex
    
cdef class SPMatrix:
    '''
    Sparse matrix base class
    '''
    cdef readonly tuple shape
    cdef readonly char *format
    cdef readonly bint isSym

    cpdef getRowCol(self, tuple key)

cdef struct llnode:
    # row node object
    llnode *prev
    llnode *next
    long col
    long idx
    
cdef struct llistdata:
    # linked list structure
    llnode **rowdata
    llnode *freenodes
    void *data
    bint isSym
    long rows, cols
    long size, sizeHint
    int itemsize
    long nnz, idx
    
cdef class LLBase(SPMatrix):
    '''
    Base class for LinkedLList sparse matrix type
    '''
    cdef readonly array data
    cdef llistdata lldat
    
    
    cdef llnode *searchNode(self, long row, long col)
    cdef llnode *getNode(self, long row, long col)
    
    cpdef dict toDOK(self, copy = ?)
    cpdef toLL(self, bint copy = ?)
    cpdef toCOO(self, bint copy = ?)
    cpdef toCSR(self, bint copy = ?)

cdef class LLB(LLBase):
    cpdef ubyte getValue(self, long idx)
    cpdef setValue(self, long idx, ubyte value)
    cpdef ubyte getitem(self, long row, long col)
    cpdef setitem(self, long row, long col, ubyte value)

cdef class LLd(LLBase):
    cpdef double getValue(self, long idx)
    cpdef setValue(self, long idx, double value)
    cpdef double getitem(self, long row, long col)
    cpdef setitem(self, long row, long col, double value)
    cpdef scale(self, double value)
    cdef cmatvec(self, double *x, double *y)
    cpdef matvec(self, double[:] x, double[:] y)

cdef class LLZ(LLBase):
    cpdef Zcomplex getValue(self, long idx)
    cpdef setValue(self, long idx, Zcomplex value)
    cpdef Zcomplex getitem(self, long row, long col)
    cpdef setitem(self, long row, long col, Zcomplex value)
    cpdef scale(self, Zcomplex value)
    cdef cmatvec(self, Zcomplex *x, Zcomplex *y)
    cpdef matvec(self, Zcomplex[:] x, Zcomplex[:] y)

cdef struct coodata:
    void *data
    long *row
    long *col
    bint isSym
    long rows, cols
    int itemsize
    long nnz

cdef class COOBase(SPMatrix):
    '''
    A sparse matrix in COOrdinate format.
    '''
    cdef readonly array data
    cdef readonly array row
    cdef readonly array col
    cdef coodata coodat
    
    cpdef dict toDOK(self, bint copy = ?)
    cpdef toLL(self, bint copy = ?)
    cpdef toCOO(self, bint copy = ?)
    cpdef toCSR(self, bint copy = ?)

cdef class COOB(COOBase):
    cpdef ubyte getValue(self, long idx)

cdef class COOd(COOBase):
    cpdef double getValue(self, long idx)
    cdef cmatvec(self, double *x, double *y)
    cpdef matvec(self, double[:] x, double[:] y)

cdef class COOZ(COOBase):
    cpdef Zcomplex getValue(self, long idx)
    cdef cmatvec(self, Zcomplex *x, Zcomplex *y)
    cpdef matvec(self, Zcomplex[:] x, Zcomplex[:] y)

cdef struct csrdata:
    void *data
    long *indptr
    long *indices
    bint isSym
    long rows, cols
    int itemsize
    long nnz

cdef class CSRBase(SPMatrix):
    cdef readonly array data
    cdef readonly array indptr
    cdef readonly array indices
    
    cdef csrdata csrdat
    
    cpdef dict toDOK(self, bint copy = ?)
    cpdef toLL(self, bint copy = ?)
    cpdef toCOO(self, bint copy = ?)
    cpdef toCSR(self, bint copy = ?)

cdef class CSRB(CSRBase):
    cpdef ubyte getValue(self, long idx)

cdef class CSRd(CSRBase):
    cpdef double getValue(self, long idx)
    cdef cmatvec(self, double *x, double *y)
    cpdef matvec(self, double[:] x, double[:] y)

cdef class CSRZ(CSRBase):
    cpdef Zcomplex getValue(self, long idx)
    cdef cmatvec(self, Zcomplex *x, Zcomplex *y)
    cpdef matvec(self, Zcomplex[:] x, Zcomplex[:] y)
    