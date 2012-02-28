# -*- coding: utf-8 -*-

class COO(object):
    '''
    Factory class for COO sparse matrix
    '''
    def __new__(self, shape, long nnz = 0, bint isSym = False, format = 'd'):
        if format == 'B':
            return COOB(shape, nnz, isSym)
        elif format == 'd':
            return COOd(shape, nnz, isSym)
        elif format == 'Z':
            return COOZ(shape, nnz, isSym)
        else:
            raise TypeError("unknown format '%s'" % format)
            
cdef class COOBase(SPMatrix):
    '''
    A sparse matrix in COOrdinate format.
    '''
    def __init__(self, shape, long nnz = 0, bint isSym = False,
                 format = None):
        SPMatrix.__init__(self, shape, format, isSym)
        
        self.coodat.rows = shape[0]
        self.coodat.cols = shape[1]
        self.coodat.isSym = isSym
        self.coodat.nnz = nnz
        self.coodat.itemsize = SIZEOF[format]
        
        if nnz > 0:
            # create new arrays of given size
            self.data = array(shape=(nnz,), itemsize=self.itemsize, format=format)
            self.coodat.data = <void *>self.data.data
            
            self.row = array(shape=(nnz,),  itemsize=sizeof(long), format="l")
            self.coodat.row = <long *>self.row.data
            
            self.col = array(shape=(nnz,),  itemsize=sizeof(long), format="l")
            self.coodat.col = <long *>self.col.data
    
    property isSym:
        def __get__(self):
            return self.coodat.isSym
            
    property nnz:
        def __get__(self):
            return self.coodat.nnz
    
    property itemsize:
        def __get__(self):
            return self.coodat.itemsize
            
    def __str__(self):
        cdef long idx
        st = ["("]

        for idx in range(self.nnz):
            args = self.coodat.row[idx], self.coodat.col[idx], self.getValue(idx)
            st.append("  (%d,%d) : %s" % args)

        st.append(")")
        return "\n".join(st)
    
    cpdef dict toDOK(self, bint copy = True):
        cdef dict ret = {}
        cdef long idx
        
        for idx in range(self.nnz):
            ret[(self.coodat.row[idx], self.coodat.col[idx])] = self.getValue(idx)
        
        return ret
    
    cpdef toLL(self, bint copy = True):
        cdef LLBase ret
        
        ret = LL(self.shape, self.nnz, self.isSym, self.format)
        COOtoLL(&self.coodat, &ret.lldat)
        
        return ret
        
    cpdef toCOO(self, bint copy = True):
        """
        Return a copy of this matrix in COOrdinate format
        """
        cdef COOBase ret
        
        if not copy:
            return self
        
        ret = self.__class__(self.shape, self.nnz, self.isSym)
        COOtoCOO(&self.coodat, &ret.coodat)
        
        return ret
    
    cpdef toCSR(self, bint copy = True):
        cdef CSRBase ret
        
        ret = CSR(self.shape, self.nnz, self.isSym, self.format)
        COOtoCSR(&self.coodat, &ret.csrdat)
        
        return ret

cdef class COOB(COOBase):
    def __init__(self, shape, long nnz = 0, bint isSym = False):
        COOBase.__init__(self, shape, nnz, isSym, 'B')
    
    cpdef ubyte getValue(self, long idx):
        return (<ubyte *>self.coodat.data)[idx]
        
cdef class COOd(COOBase):
    def __init__(self, shape, long nnz = 0, bint isSym = False):
        COOBase.__init__(self, shape, nnz, isSym, 'd')
    
    cpdef double getValue(self, long idx):
        return (<double *>self.coodat.data)[idx]
    
    cdef cmatvec(self, double *x, double *y):
        matvecCOO(&self.coodat, x, y)
    
    cpdef matvec(self, double[:] x, double[:] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        matvecCOO(&self.coodat, &x[0], &y[0])

cdef class COOZ(COOBase):
    def __init__(self, shape, long nnz = 0, bint isSym = False):
        COOBase.__init__(self, shape, nnz, isSym, 'Z')
    
    cpdef Zcomplex getValue(self, long idx):
        return (<Zcomplex *>self.coodat.data)[idx]
    
    cdef cmatvec(self, Zcomplex *x, Zcomplex *y):
        matvecCOO(&self.coodat, x, y)
    
    cpdef matvec(self, Zcomplex[:] x, Zcomplex[:] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        matvecCOO(&self.coodat, &x[0], &y[0])