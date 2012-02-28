# -*- coding: utf-8 -*-

@cython.boundscheck(False)
cdef inline int LLtoLL(llistdata *src, llistdata *dst):
    cdef llnode *root, *cur, *node
    cdef long row, idx
    
    memcpy(dst.data, src.data, src.size * src.itemsize)
    
    for row in range(src.rows):
        root = src.rowdata[row]
        
        cur = root.next
        while cur is not root:
            node = getLLNode(dst, row, cur.col)
            if node == NULL:
                return -1
            
            node.idx = cur.idx
            cur = cur.next
    
    return 0
    
@cython.boundscheck(False)
cdef inline int LLtoCOO(llistdata *src, coodata *dst):
    cdef llnode *root, *cur
    cdef long i, row
    
    i = 0
    for row in range(src.rows):
        root = src.rowdata[row]
        cur = root.next
        while cur is not root:
            dst.row[i] = row
            dst.col[i] = cur.col
            
            memcpy(dst.data + i*src.itemsize, src.data + cur.idx*src.itemsize,
                   src.itemsize)
            
            cur = cur.next
            i += 1
    return 0
@cython.boundscheck(False)
cdef inline int LLtoCSR(llistdata *src, csrdata *dst):
    cdef llnode *root, *cur
    cdef long i, row
    
    i = 0
    for row in range(src.rows):
        dst.indptr[row] = i
        root = src.rowdata[row]
        cur = root.next
        while cur is not root:
            dst.indices[i] = cur.col
            
            memcpy(dst.data + i*src.itemsize, src.data + cur.idx*src.itemsize,
                   src.itemsize)
            
            cur = cur.next
            i += 1
    
    dst.indptr[src.rows] = src.nnz
    
    return 0

@cython.boundscheck(False)
cdef inline int COOtoCOO( coodata *src, coodata *dst):
    memcpy(dst.data, src.data, src.nnz * src.itemsize)
    memcpy(dst.row, src.row, src.nnz * src.itemsize)
    memcpy(dst.col, src.col, src.nnz * src.itemsize)
    return 0
    
@cython.boundscheck(False)
cdef inline int COOtoLL(coodata *src, llistdata *dst):
    cdef llnode *node
    cdef long idx
    
    memcpy(dst.data, src.data, src.nnz * src.itemsize)
    
    for idx in range(src.nnz):
        node = getLLNode(dst, src.row[idx], src.col[idx])
        if node == NULL:
            return -1
        
        node.idx = idx
    
    return 0

@cython.boundscheck(False)
cdef inline int COOtoCSR(coodata *src, csrdata *dst):
    cdef long cumsum, row, last
    cdef long i, tmp
    
    memcpy(dst.data, src.data, src.nnz * src.itemsize)
    
    # compute number of non-zero entries per row of A
    memset(dst.indptr, 0, (src.rows + 1) * sizeof(long))
    for i in range(src.nnz):
        dst.indptr[src.row[i]] += 1
        
    # cumsum the nnz per row
    cumsum = 0
    for i in range(src.rows):
        tmp = dst.indptr[i]
        dst.indptr[i] = cumsum
        cumsum += tmp
    dst.indptr[src.rows] = src.nnz
    
    for i in range(src.nnz):
        row = src.row[i]
        dst.indices[dst.indptr[row]] = src.col[i]
        dst.indptr[row] += 1
    
    last = 0
    for i in range(src.rows + 1):
        tmp = dst.indptr[i]
        dst.indptr[i] = last
        last = tmp
    
    return 0

@cython.boundscheck(False)
cdef inline int CSRtoCSR(csrdata *src, csrdata *dst):
    memcpy(dst.data, src.data, src.nnz * src.itemsize)
    memcpy(dst.indptr, src.indptr, (src.rows + 1) * sizeof(long))
    memcpy(dst.indices, src.indices, src.nnz * sizeof(long))
    return 0

@cython.boundscheck(False)
cdef inline int CSRtoLL(csrdata *src, llistdata *dst):
    cdef llnode *root, *node
    cdef long row, j, idx
    
    memcpy(dst.data, src.data, src.nnz * src.itemsize)
    
    for row in range(src.rows):
        for j in range(src.indptr[row], src.indptr[row + 1]):
            node = appendNewLLNode(dst, row)
            if node == NULL:
                return -1
            node.col = src.indices[j]
            node.idx = j
    
    return 0

@cython.boundscheck(False)
cdef inline int CSRtoCOO(csrdata *src, coodata *dst):
    cdef long i, j
    
    memcpy(dst.data, src.data, src.nnz * src.itemsize)
    memcpy(dst.col, src.indices, src.nnz * sizeof(long))
    
    # Expand a compressed row pointer into a row array
    for i in range(src.rows):
        for j in range(src.indptr[i], src.indptr[i + 1]):
            dst.row[j] = i
            
    return 0