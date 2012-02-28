# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from libc.stdlib cimport malloc, free
from cython.view cimport array

from spmatrix cimport COOd, LLd

cdef extern:
    # arpack
    void dsaupd_(int *ido, char *bmat, int *n, char *which,
                 int *nev, double *tol, double *resid, int *ncv,
                 double *v, int *ldv, int *iparam, int *ipntr,
                 double *workd, double *workl, int *lworkl,
                 int *info)

    void dseupd_(int *rvec, char *All, int *select, double *d,
                 double *v, int *ldv, double *sigma, 
                 char *bmat, int *n, char *which, int *nev,
                 double *tol, double *resid, int *ncv, double *v,
                 int *ldv, int *iparam, int *ipntr, double *workd,
                 double *workl, int *lworkl, int *ierr)
            
cdef extern from "spooles.h":
    cdef struct factorinfo:
        pass
    
    void *spooles_factor(long *row, long *col, double *data,
                         long neq, long nnz, int symmetryflag)
    void spooles_solve(void *ptr, double *b, long neq)
    void spooles_cleanup(void *ptr)

def arpack_factor(LLd K, LLd M):
    # build matrix A = K - M
    cdef COOd A = K.toCOO()
    cdef double[:] data = A.data
    cdef long[:] row = A.row
    cdef long[:] col = A.col
    cdef long i
    
    for i in range(A.nnz):
        data[i] -= M.getitem(row[i], col[i])
    
    # factor A with spooles
    return Spooles(A)
    
def arpack(LLd K, LLd M, int nev = 3, int ncv = -1, double tol = 0., int mxiter = -1):
    cdef Spooles factor
    cdef array ret
    cdef double *resid, *workd, *workl, *z, *tmp, *d, sigma
    cdef char *bmat, *which, *howmny 
    cdef long long zsize
    cdef int neq, ido, iparam[11], ipntr[11]
    cdef int lworkl, dz, info, rvec, row, i, *select
    
    rvec = 1            # eigenvectors should be calculate
    bmat = "G"          # general eigenvalue problem    
    which = "LA"        # ask for values just to the right of shift
    howmny = "A"        # all values
    sigma = 1.          # shift value
    
    # build matrix A = K - M
    factor = arpack_factor(K, M)
    neq = factor.neq
    
    if mxiter < 0:
        mxiter = 1000
    
    if nev > neq:
        raise ValueError('nev >= neq')
        
    # Largest number of basis vectors
    if ncv < 0:
        ncv = 2 * nev + 1
    ncv = min(ncv, neq)
    
    ido = 0
    dz = neq
    iparam[1 - 1] = 1           # Shift strategy (1->exact)
    iparam[2 - 1] = 0           # Not used
    iparam[3 - 1] = mxiter      # Maximum number of iterations 
    iparam[4 - 1] = 1           # Block size 
    iparam[5 - 1] = 0           # NCONV
    iparam[6 - 1] = 0           # IUPD
    iparam[7 - 1] = 3           # shift-invert mode
    iparam[8 - 1] = 0           # NP
    iparam[9 - 1] = 0           # NUMOP
    iparam[10 - 1] = 0          # NUMOPB
    iparam[11 - 1] = 0          # NUMREO
 
    lworkl = ncv*(ncv + 8)
    info = 0
    
    resid = <double *>malloc(sizeof(double) * neq)
    if resid is NULL: 
        raise MemoryError()
    
    zsize = ncv * neq
    z = <double *>malloc(sizeof(double) * zsize)
    if z is NULL: 
        raise MemoryError() 
        
    workd = <double *>malloc(sizeof(double) * 3 * neq)
    if workd is NULL: 
        raise MemoryError() 
        
    workl = <double *>malloc(sizeof(double) * lworkl)
    if workl is NULL: 
        raise MemoryError()
    
    tmp = <double *>malloc(sizeof(double) * neq)
    if tmp is NULL: 
        raise MemoryError()
    
    d = <double *>malloc(sizeof(double) * (nev + 1))
    if d is NULL: 
        raise MemoryError()
    
    select = <int *>malloc(sizeof(int) * ncv)
    if select is NULL: 
        raise MemoryError()
    
    try:  
        while True:
            dsaupd_(&ido, bmat, &neq, which, &nev, &tol, resid, 
                    &ncv, z, &dz, iparam, ipntr, workd, workl,
                    &lworkl, &info)
                    
            for row in range(neq):
                tmp[row] = 0.
            
            if ido == -1:
                M.cmatvec(&workd[ipntr[0] - 1], tmp)
                factor.solve_(tmp)
                    
            elif ido == 1:
                for row in range(neq):
                    tmp[row] = workd[ipntr[2] - 1 + row]
                    
                factor.solve_(tmp)
                
            elif ido == 2:
                M.cmatvec(&workd[ipntr[0] - 1], tmp)
            
            elif ido == 99:
                break
                
            else:
                print "*ERROR in arpack: ido=%d\n" % ido
                return None
                
            for row in range(neq):
                workd[ipntr[1] - 1 + row] = tmp[row]
        
        if info < 0:
            print "*ERROR in arpack: info=%d\n" % info
            return None
        
        if info > 0:
            print "Warning dsaupd, info = %d\n" % info
            return None
            
        info = 0
        dseupd_(&rvec,howmny,select,d,z,&dz,&sigma,bmat,&neq,which,&nev,&tol,resid,
                &ncv,z,&dz,iparam,ipntr,workd,workl,&lworkl,&info)
        
        if info != 0:
            print "Error with dseupd, info = %d\n" % info
            return None
        
        ret = array(shape=(iparam[4],), itemsize=sizeof(double), format="d")
        for i in range(iparam[4]):
            ret[i] = d[i]
            
    finally:
        free(resid)
        free(z)
        free(workd)
        free(workl)
        free(tmp)
        free(select)
        free(d)
    
    return ret
    
cdef class Spooles:
    cdef void *ptr
    cdef readonly int neq
    
    def __init__(self, mat):
        cdef COOd coo = mat.toCOO(copy = False)
        cdef double[:] data = coo.data
        cdef long[:] row = coo.row
        cdef long[:] col = coo.col
        cdef int symflag
        
        assert coo.shape[0] == coo.shape[1], 'Expected square matrix'
        self.neq = coo.shape[0]
        
        if coo.isSym:
            symflag = 0
        else:
            symflag = 2
        
        self.ptr = spooles_factor(&row[0], &col[0], &data[0], self.neq, coo.nnz,
                                  symflag)
    
    def __dealloc__(self):
        spooles_cleanup(self.ptr)
    
    cdef solve_(self, double *b):
        spooles_solve(self.ptr, b, self.neq)
        
    cpdef solve(self, double[:] b):
        self.solve_(&b[0])