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
    def __init__(self, shape):
        dict.__init__(self)
        self.shape = shape
        
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
                elif isSym and (col,row) in self:
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
    property ndim:
        def __get__(self):
            return 2
    
    cdef matvec_(self, double *x, double *y):
        cdef double[:] y_view, data_view = self.data
        cdef int[:] row_view = self.row
        cdef int[:] col_view = self.col
        cdef int i
        
        for i in range(self.nnz):
            y[row_view[i]] += data_view[i] * x[col_view[i]]
    
    cpdef matvec(self, object[double, ndim=1] x, object[double, ndim=1] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        self.matvec_(&x[0], &y[0])
    
    cpdef DOK toDOK(self, bint copy = True):
        cdef DOK ret = DOK(self.shape)
        cdef double[:] y_view, data_view = self.data
        cdef int[:] row_view = self.row
        cdef int[:] col_view = self.col
        cdef int i
        
        for i in range(self.nnz):
            ret[(row_view[i], col_view[i])] = data_view[i]
            
        return ret
    
    cpdef COO toCOO(self, bint copy = True):
        """
        Return a copy of this matrix in COOrdinate format
        """
        cdef double[:] data_view = self.data
        cdef int[:] row_view = self.row
        cdef int[:] col_view = self.col
        cdef rows, cols
        
        if not copy:
            return self
            
        rows, cols = self.shape
        ret = COO()
        
        ret.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
        cdef double[:] tmpd_view = ret.data
        tmpd_view[:] = data_view
        
        ret.row = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        cdef int[:] tmpi_view = ret.row
        tmpi_view[:] = row_view
        
        ret.col = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        tmpi_view = ret.col
        tmpi_view[:] = col_view
        
        ret.shape = rows, cols
        ret.nnz = self.nnz
        
        return ret
        
    cpdef CSR toCSR(self, bint copy = True):
        """
        Return a copy of this matrix in Compressed Sparse Row format
        """
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
    property ndim:
        def __get__(self):
            return 2
    
    cdef matvec_(self, double *x, double *y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        cdef double[:] y_view, data_view = self.data
        cdef int[:] indptr_view = self.indptr
        cdef int[:] indices_view = self.indices
        cdef double tmp
        cdef int i, j, rows, cols
        
        rows, cols = self.shape
        
        for i in range(rows):
            tmp = y[i]
            for j in range(indptr_view[i], indptr_view[i + 1]):
                tmp += data_view[j] * x[indices_view[j]]
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
        cdef COO ret = COO()
        cdef double[:] tmpd_view, data_view = self.data
        cdef int[:] tmpi_view, indptr_view = self.indptr
        cdef int[:] indices_view = self.indices
        cdef int i, j, rows, cols
        
        rows, cols = self.shape
        
        if copy:
            ret.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
            tmpd_view = ret.data
            tmpd_view[:] = data_view
            
            ret.col = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
            tmpi_view = ret.col
            tmpi_view[:] = indices_view
        else:
            ret.data = self.data
            ret.col = self.indices
        
        ret.row = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        tmpi_view = ret.row
        
        # Expand a compressed row pointer into a row array
        for i in range(rows):
            for j in range(indptr_view[i], indptr_view[i + 1]):
                tmpi_view[j] = i
        
        ret.shape = rows, cols
        ret.nnz = self.nnz
        
        return ret
    
    cpdef CSR toCSR(self, bint copy = True):
        """
        Return a copy of this matrix in Compressed Sparse Row format
        """
        cdef CSR ret = CSR()
        cdef double[:] tmpd_view, data_view = self.data
        cdef int[:] tmpi_view, indptr_view = self.indptr
        cdef int[:] indices_view = self.indices
        cdef int rows, cols
        
        if not copy:
            return self
            
        rows, cols = self.shape
        
        ret.data = array(shape=(self.nnz,), itemsize=sizeof(double), format="d")
        tmpd_view = ret.data
        tmpd_view[:] = data_view
        
        ret.indices = array(shape=(self.nnz,), itemsize=sizeof(int), format="i")
        tmpi_view = ret.indices
        tmpi_view[:] = indices_view
        
        ret.indptr = array(shape=(self.shape[0] + 1,), itemsize=sizeof(int), format="i")
        tmpi_view = ret.indptr
        tmpi_view[:] = indptr_view
        
        ret.shape = rows, cols
        ret.nnz = self.nnz
        
        return ret