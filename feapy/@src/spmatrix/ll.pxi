# -*- coding: utf-8 -*-

class LL(object):
    '''
    Factory class for LL sparse matrix
    '''
    def __new__(self, shape, long sizeHint = 0, bint isSym = False,
                format = 'd'):
        if format == 'B':
            return LLB(shape, sizeHint, isSym)
        elif format == 'd':
            return LLd(shape, sizeHint, isSym)
        elif format == 'Z':
            return LLZ(shape, sizeHint, isSym)
        else:
            raise TypeError("unknown format '%s'" % format)
            
cdef class LLBase(SPMatrix):
    '''
    Base class for LinkedLList sparse matrix type
    '''
    def __init__(self, shape, long sizeHint = 0, bint isSym = False,
                 format = None):
        SPMatrix.__init__(self, shape, format, isSym)
        
    def __cinit__(self, shape, long sizeHint = 0, bint isSym = False,
                  format = None):
        cdef llnode *root, *node, *next
        cdef long i
        
        # setup data structure
        self.lldat.isSym = isSym
        self.lldat.rows = shape[0]
        self.lldat.cols = shape[1]
        self.lldat.size = max(1, sizeHint)
        self.lldat.sizeHint = max(1, sizeHint)
        self.lldat.itemsize = SIZEOF.get(format, 0)
        self.lldat.nnz = 0
        self.lldat.idx = 0
    
        # setup row structure
        self.lldat.rowdata = <llnode **>malloc(self.lldat.rows * sizeof(llnode *))
        if self.lldat.rowdata == NULL:
            raise MemoryError('Could not allocate memory')
        
        for i in range(self.lldat.rows):
            root = createLLNode()
            if root == NULL:
                raise MemoryError('Could not allocate memory')
            
            root.prev = root
            root.next = root
            self.lldat.rowdata[i] = root
        
        # fill up free list
        root = self.lldat.freenodes = createLLNode()
        if root == NULL:
            raise MemoryError('Could not allocate memory')
        root.prev = root
        root.next = root
        
        for i in range(self.lldat.size):
            node = createLLNode(NULL, NULL, -1, self.lldat.idx)
            if node == NULL:
                raise MemoryError('Could not allocate memory')
            
            # append to end
            appendLLNode(root, node)
            
            # increase value index
            self.lldat.idx += 1
            
    def __dealloc__(self):
        cdef llnode *root, *next
        cdef long i
        
        destroyLLNode(self.lldat.freenodes)
        
        # free row structure
        if self.lldat.rowdata != NULL:
            try:
                for i in range(self.lldat.rows):
                    destroyLLRow(self.lldat.rowdata[i])
            finally:
                free(self.lldat.rowdata)
        
        if self.lldat.data != NULL:
            free(self.lldat.data)
                
    property isSym:
        def __get__(self):
            return self.lldat.isSym
            
    property nnz:
        def __get__(self):
            return self.lldat.nnz
    
    property itemsize:
        def __get__(self):
            return self.lldat.itemsize
    
    cdef llnode *searchNode(self, long row, long col):
        '''Search for node in row'''
        return searchLLNode(&self.lldat, row, col)
        
    cdef llnode *getNode(self, long row, long col):
        '''get existing or create new node'''
        cdef llnode *ret = getLLNode(&self.lldat, row, col)
        
        if ret != NULL:
            # sync array
            self.data.data = <char *>self.lldat.data
        
        return ret
        
    def __getitem__(self, tuple key):
        cdef llnode *node
        cdef long row, col
        
        row, col = self.getRowCol(key)
        
        node = self.searchNode(row, col)
        if node != NULL:
            return self.getValue(node.idx)
        else:
            return self.getValue(-1)
            
    def __setitem__(self, tuple key, value):
        cdef llnode *node
        cdef long row, col
        
        row, col = self.getRowCol(key)
        
        node = self.getNode(row, col)
        if node != NULL:
            self.setValue(node.idx, value)
        else:
            raise SPMatrixError('failed to create node')
    
    def __str__(self):
        cdef llnode *root, *cur
        cdef long row
        st = ["("]
        
        for row in range(self.shape[0]):
            root = self.lldat.rowdata[row]
            cur = root.next
            while cur is not root:
                args = row, cur.col, self.getValue(cur.idx)
                st.append("  (%d,%d) : %s" % args)
                cur = cur.next
        
        st.append(")")
        return "\n".join(st)
        
    @cython.boundscheck(False)
    cpdef dict toDOK(self, copy = True):
        cdef dict ret = dict()
        cdef llnode *root, *cur
        cdef long row
        
        for row in range(self.shape[0]):
            root = self.lldat.rowdata[row]
            cur = root.next
            while cur is not root:
                ret[row, cur.col] = self.getValue(cur.idx)
                cur = cur.next
            
        return ret
    
    cpdef toLL(self, bint copy = True):
        cdef LLBase ret
        
        if not copy:
            return self
        
        ret = self.__class__(self.shape, self.lldat.size, self.lldat.isSym)
        if LLtoLL(&self.lldat, &ret.lldat) == -1:
            raise MemoryError('Could not copy matrix')
        
        return ret
    
    cpdef toCOO(self, bint copy = True):
        cdef COOBase ret
        
        ret = COO(self.shape, self.nnz, self.lldat.isSym, self.format)
        if LLtoCOO(&self.lldat, &ret.coodat) == -1:
            raise MemoryError('Could not copy matrix')
        
        return ret
    
    cpdef toCSR(self, bint copy = True):
        cdef CSRBase ret
        
        ret = CSR(self.shape, self.nnz, self.lldat.isSym, self.format)
        if LLtoCSR(&self.lldat, &ret.csrdat) == -1:
            raise MemoryError('Could not copy matrix')
        
        return ret

cdef class LLB(LLBase):
    def __init__(self, shape, long sizeHint = 0, bint isSym = False):
        LLBase.__init__(self, shape, sizeHint, isSym, 'B')
    
    def __cinit__(self, shape, long sizeHint = 0, bint isSym = False):
        # set data type
        self.lldat.itemsize = sizeof(ubyte)
        
        # alloc initial data
        self.lldat.data = malloc(self.lldat.size * self.lldat.itemsize)
        self.data.data = <char *>self.lldat.data
        if self.lldat.data == NULL:
            raise MemoryError('Could not allocate memory')
            
        self.data = array(shape=(self.lldat.size,), itemsize=self.lldat.itemsize,
                          format="B", allocate_buffer=False)
    
    cpdef ubyte getValue(self, long idx):
        if idx > -1:
            return (<ubyte *>self.lldat.data)[idx]
        else:
            return 0
            
    cpdef setValue(self, long idx, ubyte value):
        (<ubyte *>self.lldat.data)[idx] = value
        
cdef class LLd(LLBase):
    def __init__(self, shape, long sizeHint = 0, bint isSym = False):
        LLBase.__init__(self, shape, sizeHint, isSym, 'd')
    
    def __cinit__(self, shape, long sizeHint = 0, bint isSym = False):
        # set data type
        self.lldat.itemsize = sizeof(double)
        
        # alloc initial data
        self.lldat.data = malloc(self.lldat.size * self.lldat.itemsize)
        self.data.data = <char *>self.lldat.data
        if self.lldat.data == NULL:
            raise MemoryError('Could not allocate memory')
            
        self.data = array(shape=(self.lldat.size,), itemsize=self.lldat.itemsize,
                          format="d", allocate_buffer=False)
    
    cpdef double getValue(self, long idx):
        if idx > -1:
            return (<double *>self.lldat.data)[idx]
        else:
            return 0.
            
    cpdef setValue(self, long idx, double value):
        (<double *>self.lldat.data)[idx] = value
    
    cpdef scale(self, double value):
        scaleLL(&self.lldat, value)
        return self
    
    cdef cmatvec(self, double *x, double *y):
        matvecLL(&self.lldat, x, y)
                
    cpdef matvec(self, double[:] x, double[:] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        matvecLL[double](&self.lldat, &x[0], &y[0])

cdef class LLZ(LLBase):
    def __init__(self, shape, long sizeHint = 0, bint isSym = False):
        LLBase.__init__(self, shape, sizeHint, isSym, 'Z')
    
    def __cinit__(self, shape, long sizeHint = 0, bint isSym = False):
        # set data type
        self.lldat.itemsize = sizeof(Zcomplex)
        
        # alloc initial data
        self.lldat.data = malloc(self.lldat.size * self.lldat.itemsize)
        self.data.data = <char *>self.lldat.data
        if self.lldat.data == NULL:
            raise MemoryError('Could not allocate memory')
        
        self.data = array(shape=(self.lldat.size,), itemsize=self.lldat.itemsize,
                          format="d", allocate_buffer=False)
    
    cpdef Zcomplex getValue(self, long idx):
        if idx > -1:
            return (<Zcomplex *>self.lldat.data)[idx]
        else:
            return 0.
            
    cpdef setValue(self, long idx, Zcomplex value):
        (<Zcomplex *>self.lldat.data)[idx] = value
    
    cpdef scale(self, Zcomplex value):
        scaleLL(&self.lldat, value)
        return self
    
    cdef cmatvec(self, Zcomplex *x, Zcomplex *y):
        matvecLL(&self.lldat, x, y)
                
    cpdef matvec(self, Zcomplex[:] x, Zcomplex[:] y):
        '''
        Multiply matrix with dense vector:
            y += M * x
        '''
        matvecLL(&self.lldat, &x[0], &y[0])
        
cdef inline llnode *createLLNode(llnode *prev = NULL, llnode *next = NULL,
                                long col = -1, long idx = -1):
    '''Create new unlinked node'''
    cdef llnode *ret
    
    ret = <llnode *>malloc(sizeof(llnode))
    if ret == NULL:
        return ret
        
    ret.prev = prev
    ret.next = next
    ret.col = col
    ret.idx = idx
    
    return ret

cdef  inline int destroyLLNode(llnode *node):
    '''Destroy single node'''
    if node != NULL:
        free(node)

cdef inline destroyLLRow(llnode *root):
    '''Destroy all nodes in row'''
    cdef llnode *cur, *next
    
    cur = root.next
    while cur is not root:
        next = cur.next
        destroyLLNode(cur)
        cur = next
    
    destroyLLNode(root)

@cython.boundscheck(False)
cdef inline insertLLNodeLeft(llnode *cur, llnode *node):
    '''insert node left to cur'''
    cdef llnode *prev = cur.prev
    prev.next =  node
    node.prev = prev
    node.next = cur
    cur.prev =  node

@cython.boundscheck(False)
cdef inline appendLLNode(llnode *root, llnode *node):
    '''append node at end of row'''
    cdef llnode *cur = root.prev
    cur.next = root.prev = node
    node.prev = cur
    node.next = root

@cython.boundscheck(False)
cdef inline llnode *appendNewLLNode(llistdata *self, long row):
    '''create new node and append to end'''
    cdef llnode *root = self.rowdata[row]
    cdef llnode *node = createLLNode(NULL, NULL, -1, self.idx)
    if node == NULL:
        return NULL
    
    # increase amount of data if needed
    if LLRealloc(self, 1) == -1:
        return NULL
    
    # append to end
    appendLLNode(root, node)
        
    self.idx += 1
    self.nnz += 1
    
    return node
    
@cython.boundscheck(False)
cdef inline removeLLNode(llnode *node):
    '''remove node from list'''
    cdef llnode *prev, *next
    prev, next = node.prev, node.next
    prev.next = next
    next.prev = prev

@cython.boundscheck(False)
cdef inline llnode *searchLLNode(llistdata *self, long row, long col):
    '''search for node in row'''
    cdef llnode *root = self.rowdata[row]
    cdef llnode *node, *cur, *prev, *next
    
    # check for access above diagonal for symetric matrices
    if self.isSym and row < col:
        return NULL

    # check for empty row
    if root.next is root and root.prev is root:
        return NULL
    # check if we are inside bounds
    elif root.next.col > col or root.prev.col < col:
        return NULL
    
    # find fastest search direction
    if col - root.next.col < root.prev.col - col:
        cur = root.next
        while cur is not root:
            if cur.col == col:
                break
            cur = cur.next
        else:
            return NULL
    else:
        cur = root.prev
        while cur is not root:
            if cur.col == col:
                break
            cur = cur.prev
        else:
            return NULL
    # found
    return cur

@cython.boundscheck(False)
cdef inline llnode *getLLNode(llistdata *self, long row, long col):
    '''
    Get existing node or create new node
    '''
    cdef llnode *root = self.rowdata[row]
    cdef llnode *node, *cur, *prev, *next
    cdef bint inside, empty
    
    # check for access above diagonal for symetric matrices
    if self.isSym and row < col:
        return NULL
        
    empty = False
    inside = True
    
    # check for empty row
    if root.next is root and root.prev is root:
        empty = True
    elif root.next.col > col or root.prev.col < col:
        # check if we are inside bounds
        inside = False
    
    if empty:
        # default to append to end
        cur = root
    else:
        if inside:
            # find position from first node
            if col - root.next.col < root.prev.col - col:
                cur = root.next
                while cur is not root:
                    if cur.col >= col:
                        break
                    cur = cur.next
            else:
                cur = root.prev
                while cur is not root:
                    if cur.col <= col:
                        # right to current
                        if cur.col != col:
                            cur = cur.next
                        break
                    cur = cur.prev
                    
            # found node
            if cur.col == col:
                return cur
        else:
            if root.next.col >= col:
                cur = root.next
            else:
                cur = root
    
    if self.freenodes.prev.idx != -1:
        # reuse last node
        node = self.freenodes.prev
        
        # remove from free list
        removeLLNode(node)
        
        # update node col
        node.col = col
    else:
        # new value
        node = createLLNode(NULL, NULL, col, self.idx)
        if node == NULL:
            return NULL
        
        # increase amount of data
        if LLRealloc(self, 1) == -1:
            return NULL
        
        self.idx += 1
    
    self.nnz += 1
    
    if cur is not root:
        # insert left
        insertLLNodeLeft(cur, node)
    else:
        # append to end
        appendLLNode(root, node)
    
    return node

@cython.boundscheck(False)
cdef inline int LLRealloc(llistdata *self, long items):
    '''
    Adjust data memory if needed. Return -1 if
    an error occured or return size adjustment
    '''
    cdef void *tmp
    
    if self.idx + items > self.size:
        self.sizeHint = <long>(1.5*self.sizeHint) + 1
        self.size += self.sizeHint
        
        tmp = <void *>realloc(self.data, self.size * self.itemsize)
        if tmp == NULL:
            return -1
            
        self.data = tmp
        
        return items
        
    return 0