# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from cython cimport array

cdef class DOK(dict):
    '''
    Dictionary Of Keys based sparse matrix.

    This is an efficient structure for constructing sparse
    matrices incrementally.
    
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    '''
    property nnz:
        def __get__(self):
            return dict.__len__(self)
    
    property ndim:
        def __get__(self):
            return 2
            
    def  __getitem__(self, tuple key):
        cdef size_t row, col
        
        try:
            row, col = key
        except (ValueError, TypeError):
            raise TypeError('index must be a pair of integers')
        
        if row < 0 or col < 0:
            raise IndexError('index out of bounds')
            
        return dict.get(self, (row, col), 0.)
        
    def __setitem__(self, tuple key, double value):
        cdef size_t row, col
        
        try:
            row, col = key
        except (ValueError, TypeError):
            raise TypeError, "index must be a pair of integers"
        
        if row < 0 or col < 0:
            raise IndexError('index out of bounds')
            
        if value == 0.:
            if (row, col) in self:
                del self[(row, col)]
        else:
            dict.__setitem__(self, (row, col), value)
    
    cpdef COO toCOO(self):
        """ Return a copy of this matrix in COOrdinate format"""
        cdef COO ret = COO()
        
        ret.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
        cdef double[:] data_view = ret.data
        
        ret.row = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        cdef int[:] row_view = ret.row
        
        ret.col = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        cdef int[:] col_view = ret.col
        
        cdef int i = 0
        cdef int rows = 0,cols = 0
        for key in sorted(self):
            value = self[key]
            data_view[i] = value
            
            row_view[i] = key[0]
            rows = max(row_view[i], rows)
            
            col_view[i] = key[1]
            cols = max(col_view[i], cols)
            
            i += 1
        
        ret.nnz = self.nnz
        ret.shape = (rows + 1,cols + 1)
        
        return ret
        
    cpdef CSR toCSR(self):
        """ Return a copy of this matrix in Compressed Sparse Row format"""
        return self.toCOO().toCSR()
    
    cpdef CSC toCSC(self):
        """ Return a copy of this matrix in Compressed Sparse Column format"""
        return self.toCOO().toCSC()
        
cdef class COO:
    '''
    A sparse matrix in COOrdinate format.
    
    Also known as the 'ijv' or 'triplet' format.

    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix
    '''
    cdef public array data
    cdef public array row
    cdef public array col
    cdef readonly int nnz
    cdef readonly tuple shape
    
    property ndim:
        def __get__(self):
            return 2
    
    cpdef CSC toCSC(self):
        """ Return a copy of this matrix in Compressed Sparse Column format"""
        cdef CSC ret = CSC()
        cdef CSR csr
        try:
            self.row, self.col = self.col, self.row
            rows, cols = self.shape
            self.shape = cols, rows
            
            csr = self.toCSR()
        finally:
            self.row, self.col = self.col, self.row
            rows, cols = self.shape
            self.shape = cols, rows
        
        ret.data = csr.data
        ret.indptr = csr.indptr
        ret.indices = csr.indices
        ret.nnz = csr.nnz
        ret.shape = csr.shape
        
        return ret
        
    cpdef CSR toCSR(self):
        """ Return a copy of this matrix in Compressed Sparse Row format"""
        cdef CSR ret = CSR()
        
        cdef int i, rows, cols
        rows, cols = self.shape
        
        cdef double[:] data_view = self.data
        cdef int[:] row_view = self.row
        cdef int[:] col_view = self.col
        
        ret.indptr = array(shape=(rows + 1,), itemsize=sizeof(int), format="i")
        cdef int[:] indptr_view = ret.indptr
        
        ret.indices = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        cdef int[:] indices_view = ret.indices
        
        ret.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
        cdef double[:] resdata_view = ret.data
        
        # compute number of non-zero entries per row of A
        indptr_view[:] = 0
        for i in range(self.nnz):
            indptr_view[row_view[i]] += 1
        
        # cumsum the nnz per row to get Bp[]
        cdef int tmp, cumsum = 0
        for i in range(rows):
            tmp = indptr_view[i]
            indptr_view[i] = cumsum
            cumsum += tmp
        indptr_view[rows] = self.nnz
        
        # write Aj,Ax into Bj,Bx
        cdef int row, dest
        for i in range(self.nnz):
            row = row_view[i]
            dest = indptr_view[row]
            
            indices_view[dest] = col_view[i]
            resdata_view[dest] = data_view[i]
            
            indptr_view[row] += 1
        
        cdef int last = 0
        for i in range(rows + 1):
            tmp = indptr_view[i]
            indptr_view[i] = last
            last = tmp
        
        ret.shape = rows, cols
        ret.nnz = self.nnz
        
        return ret

cdef class CSR:
    '''
    Compressed Sparse Row matrix
    
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        CSR format data array of the matrix
    indices
        CSR format index array of the matrix
    indptr
        CSR format index pointer array of the matrix
    '''
    cdef public array data
    cdef public array indptr
    cdef public array indices
    cdef readonly int nnz
    cdef readonly tuple shape
    
    property ndim:
        def __get__(self):
            return 2

cdef class CSC:
    '''
    Compressed Sparse Column matrix
    
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        Data array of the matrix
    indices
        CSC format index array
    indptr
        CSC format index pointer array
    '''
    cdef public array data
    cdef public array indptr
    cdef public array indices
    cdef readonly int nnz
    cdef readonly tuple shape
    
    property ndim:
        def __get__(self):
            return 2
        