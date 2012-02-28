# -*- coding: utf-8 -*-

class CSR(object):
    '''
    Factory class for CSR sparse matrix
    '''
    def __new__(self, shape, long nnz = 0, bint isSym = False,
                format = 'd'):
        if format == 'B':
            return CSRB(shape, nnz, isSym)
        elif format == 'd':
            return CSRd(shape, nnz, isSym)
        elif format == 'Z':
            return CSRZ(shape, nnz, isSym)
        else:
            raise TypeError("unknown format '%s'" % format)
            
cdef class CSRBase(SPMatrix):
    def __init__(self, shape, long nnz = 0, bint isSym = False,
                 format = None):
        SPMatrix.__init__(self, shape, format, isSym)
        
        self.csrdat.rows = shape[0]
        self.csrdat.cols = shape[1]
        self.csrdat.isSym = isSym
        self.csrdat.nnz = nnz
        self.csrdat.itemsize = SIZEOF[format]
        
        if nnz > 0:
            # create new arrays of given size
            self.data = array(shape=(nnz,), itemsize=self.itemsize, format=format)
            self.csrdat.data = <void *>self.data.data
            
            self.indptr = array(shape=(shape[0] + 1,), itemsize=sizeof(long), format="l")
            self.csrdat.indptr = <long *>self.indptr.data
            
            self.indices = array(shape=(nnz,), itemsize=sizeof(long), format="l")
            self.csrdat.indices = <long *>self.indices.data
    
    property isSym:
        def __get__(self):
            return self.csrdat.isSym
            
    property nnz:
        def __get__(self):
            return self.csrdat.nnz
    
    property itemsize:
        def __get__(self):
            return self.csrdat.itemsize
    
    def __str__(self):
        cdef long i, j
        
        st = ["("]
        
        for i in range(self.csrdat.rows):
            for j in range(self.csrdat.indptr[i], self.csrdat.indptr[i + 1]):
                args = i, self.csrdat.indices[j], self.getValue(j)
                st.append("  (%d,%d) : %s" % args)

        st.append(")")
        return "\n".join(st)
    
    cpdef dict toDOK(self, bint copy = True):
        cdef dict ret = {}
        cdef long i, j
        
        for i in range(self.csrdat.rows):
            for j in range(self.csrdat.indptr[i], self.csrdat.indptr[i + 1]):
                ret[(i, self.csrdat.indices[j])] = self.getValue(j)
        
        return ret
    
    cpdef toLL(self, bint copy = True):
        cdef LLBase ret
        
        ret = LL(self.shape, self.nnz, self.isSym, self.format)
        
        CSRtoLL(&self.csrdat, &ret.lldat)
        
        return ret
    
    cpdef toCOO(self, bint copy = True):
        cdef COOBase ret
        
        ret = COO(self.shape, self.nnz, self.isSym, self.format)
        
        CSRtoCOO(&self.csrdat, &ret.coodat)
        
        return ret
    
    cpdef toCSR(self, bint copy = True):
        """
        Return a copy of this matrix in CSR format
        """
        cdef CSRBase ret
        
        if not copy:
            return self
        
        ret = self.__class__(self.shape, self.nnz, self.isSym)
        
        CSRtoCSR(&self.csrdat, &ret.csrdat)
        
        return ret

cdef class CSRB(CSRBase):
    def __init__(self, shape, long nnz = 0, bint isSym = False):
        CSRBase.__init__(self, shape, nnz, isSym, 'B')
    
    cpdef ubyte getValue(self, long idx):
        return (<ubyte *>self.csrdat.data)[idx]
        
cdef class CSRd(CSRBase):
    def __init__(self, shape, long nnz = 0, bint isSym = False):
        CSRBase.__init__(self, shape, nnz, isSym, 'd')
    
    cpdef double getValue(self, long idx):
        return (<double *>self.csrdat.data)[idx]
    
    cdef cmatvec(self, double *x, double *y):
        matvecCSR(&self.csrdat, x, y)
    
    cpdef matvec(self, double[:] x, double[:] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        matvecCSR(&self.csrdat, &x[0], &y[0])

cdef class CSRZ(CSRBase):
    def __init__(self, shape, long nnz = 0, bint isSym = False):
        CSRBase.__init__(self, shape, nnz, isSym, 'Z')
    
    cpdef Zcomplex getValue(self, long idx):
        return (<Zcomplex *>self.csrdat.data)[idx]
    
    cdef cmatvec(self, Zcomplex *x, Zcomplex *y):
        matvecCSR(&self.csrdat, x, y)
    
    cpdef matvec(self, Zcomplex[:] x, Zcomplex[:] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        matvecCSR(&self.csrdat, &x[0], &y[0])