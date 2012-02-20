# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
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
    def __init__(self, shape, data = None):
        dict.__init__(self)
        
        if not data is None:
            self.update(data)
            
        self.shape = shape
        
    property nnz:
        def __get__(self):
            return dict.__len__(self)
    
    property ndim:
        def __get__(self):
            return 2
    
    property dtype:
        def __get__(self):
            return float
            
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
    
    cpdef DOK toDOK(self, bint copy = True):
        '''
        Return a copy of this matrix in DOK format
        '''
        if copy:
            ret = DOK(self.shape)
            ret.update(self)
            return ret
        else:
            return self
    
    cdef matvec_(self, double *x, double *y, bint isSym = False):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        cdef double s
        cdef int row, col, rows, cols
        
        rows, cols = self.shape
        
        for row in range(rows):
            s = 0.
            for col in range(cols):
                if (row,col) in self:
                    s += self[row,col] * x[col]
                elif isSym:
                    if col > row and (col,row) in self:
                        s += self[col,row] * x[col]
            
            y[row] += s
        
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y, bint isSym = False):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        self.matvec_(&x[0], &y[0], isSym)
        
    cpdef COO toCOO(self, bint copy = True):
        """
        Return a copy of this matrix in COOrdinate format
        """
        cdef COO ret = COO(None, self.nnz)
        
        cdef int i = 0
        cdef int rows = 0,cols = 0
        for key in sorted(self):
            value = self[key]
            ret.dataptr[i] = value
            
            ret.rowptr[i] = key[0]
            rows = max(ret.rowptr[i], rows)
            
            ret.colptr[i] = key[1]
            cols = max(ret.colptr[i], cols)
            
            i += 1
        
        ret.shape = (rows + 1,cols + 1)
        
        return ret
        
    cpdef CSR toCSR(self, bint copy = True):
        """ Return a copy of this matrix in Compressed Sparse Row format"""
        return self.toCOO().toCSR()
        
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
    def __init__(self, shape, int nnz = 0):
        self.shape = shape
        self.nnz = nnz
        
        if nnz > 0:
            # create new empty array
            self.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
            self.dataptr = <double *>self.data.data
            
            self.row = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
            self.rowptr = <int *>self.row.data
            
            self.col = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
            self.colptr = <int *>self.col.data
        
    property dtype:
        def __get__(self):
            return float
    
    property ndim:
        def __get__(self):
            return 2
    
    cdef matvec_(self, double *x, double *y):
        cdef int i
        for i in range(self.nnz):
            y[self.rowptr[i]] += self.dataptr[i] * x[self.colptr[i]]
    
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        self.matvec_(&x[0], &y[0])
    
    cpdef DOK toDOK(self, bint copy = True):
        cdef DOK ret = DOK(self.shape)
        cdef int i
        
        for i in range(self.nnz):
            ret[(self.rowptr[i], self.colptr[i])] = self.dataptr[i]
        
        return ret
    
    cpdef COO toCOO(self, bint copy = True):
        """
        Return a copy of this matrix in COOrdinate format
        """
        cdef COO ret
        
        if not copy:
            return self
        
        ret = COO(self.shape, self.nnz)
        
        memcpy(ret.dataptr, self.dataptr, self.nnz * sizeof(double))
        memcpy(ret.rowptr, self.rowptr, self.nnz * sizeof(int))
        memcpy(ret.colptr, self.colptr, self.nnz * sizeof(int))
        
        return ret
        
    cpdef CSR toCSR(self, bint copy = True):
        """
        Return a copy of this matrix in Compressed Sparse Row format
        """
        cdef CSR ret
        cdef int i, rows, cols
        rows, cols = self.shape
        
        if copy:
            ret = CSR(self.shape, self.nnz)
            memcpy(ret.ix, self.dataptr, self.nnz * sizeof(double))
        
        else:
            ret = CSR(self.shape)
            ret.nnz = self.nnz
            
            # share data
            ret.data = self.data
            ret.ix = self.dataptr
            
            ret.indptr = array(shape=(rows + 1,), itemsize=sizeof(int), format="i")
            ret.ia = <int *>ret.indptr.data
            
            ret.indices = array(shape=(ret.nnz,), itemsize=sizeof(int), format="i")
            ret.ja = <int *>ret.indices.data
        
        # compute number of non-zero entries per row of A
        memset(ret.ia, 0, (rows + 1) * sizeof(int))
        for i in range(self.nnz):
            ret.ia[self.rowptr[i]] += 1
        
        # cumsum the nnz per row to get Bp[]
        cdef int tmp, cumsum = 0
        for i in range(rows):
            tmp = ret.ia[i]
            ret.ia[i] = cumsum
            cumsum += tmp
        ret.ia[rows] = self.nnz
        
        cdef int row
        for i in range(self.nnz):
            row = self.rowptr[i]
            ret.ja[ret.ia[row]] = self.colptr[i]
            ret.ia[row] += 1
        
        cdef int last = 0
        for i in range(rows + 1):
            tmp = ret.ia[i]
            ret.ia[i] = last
            last = tmp
        
        ret.shape = rows, cols
        ret.nnz = self.nnz
        
        return ret

cdef class CSR:
    '''
    Compressed Sparse Row matrix
    
    CSR format is prefered for arithmetic operations.
    
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
    def __init__(self, shape, int nnz = 0):
        self.shape = shape
        self.nnz = nnz
        
        if nnz > 0:
            # create new empty array
            self.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
            self.ix = <double *>self.data.data
            
            self.indptr = array(shape=(shape[0] + 1,), itemsize=sizeof(int), format="i")
            self.ia = <int *>self.indptr.data
            
            self.indices = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
            self.ja = <int *>self.indices.data
    
    property dtype:
        def __get__(self):
            return float
            
    property ndim:
        def __get__(self):
            return 2
    
    cdef matvec_(self, double *x, double *y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        cdef double tmp
        cdef int i, j, rows, cols
        
        rows, cols = self.shape
        
        for i in range(rows):
            tmp = y[i]
            for j in range(self.ia[i], self.ia[i + 1]):
                tmp += self.ix[j] * x[self.ja[j]]
            y[i] = tmp
        
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        self.matvec_(&x[0], &y[0])
    
    cpdef DOK toDOK(self, bint copy = True):
        '''
        Return a copy of this matrix in DOK format
        '''
        return self.toCOO(copy = False).toDOK()
        
    cpdef COO toCOO(self, bint copy = True):
        """
        Return a copy of this matrix in COOrdinate format
        """
        cdef COO ret
        cdef int i, j, rows, cols
        rows, cols = self.shape
        
        if copy:
            ret = COO(self.shape, self.nnz)
            memcpy(ret.dataptr, self.ix, self.nnz * sizeof(double))
            memcpy(ret.colptr, self.ja, self.nnz * sizeof(int))
        else:
            ret = COO(self.shape)
            ret.nnz = self.nnz
            
            ret.data = self.data
            ret.dataptr = self.ix
            
            ret.row = array(shape=(ret.nnz,), itemsize=sizeof(int), format="i")
            ret.rowptr = <int *>ret.row.data
            
            ret.col = self.indices
            ret.colptr = self.ja
        
        # Expand a compressed row pointer into a row array
        for i in range(rows):
            for j in range(self.ia[i], self.ia[i + 1]):
                ret.rowptr[j] = i
        
        return ret
    
    cpdef CSR toCSR(self, bint copy = True):
        """
        Return a copy of this matrix in Compressed Sparse Row format
        """
        cdef CSR ret
        
        if not copy:
            return self
        
        ret = CSR(self.shape, self.nnz)
        memcpy(ret.ix, self.ix, self.nnz * sizeof(double))
        memcpy(ret.ia, self.ia, (self.shape[0] + 1) * sizeof(int))
        memcpy(ret.ja, self.ja, self.nnz * sizeof(int))
        
        return ret