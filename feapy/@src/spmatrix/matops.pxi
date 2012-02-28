# -*- coding: utf-8 -*-

@cython.boundscheck(False)
cdef inline scaleLL(llistdata *self, numeric value):
    cdef numeric *data = <numeric *>self.data
    cdef long i
    
    for i in range(self.size):
        data[i] *= value

@cython.boundscheck(False)
cdef inline matvecLL(llistdata *self, numeric *x, numeric *y):
    cdef llnode *root, *cur
    cdef numeric *data = <numeric *>self.data
    cdef numeric value
    cdef long row
    
    if self.isSym:
        for row in range(self.rows):
            root = self.rowdata[row]
            cur = root.next
            while cur is not root:
                value = data[cur.idx]
                y[row] += value * x[cur.col]
                
                if row != cur.col:
                    y[cur.col] += value * x[row]
                    
                cur = cur.next
    else:
        for row in range(self.rows):
            root = self.rowdata[row]
            cur = root.next
            while cur is not root:
                y[row] += data[cur.idx] * x[cur.col]
                cur = cur.next
        
@cython.boundscheck(False)
cdef inline matvecCOO(coodata *self, numeric *x, numeric *y):
    cdef numeric *data = <numeric *>self.data
    cdef long i, row, col
    
    if self.isSym:
        for i in range(self.nnz):
            row = self.row[i]
            col = self.col[i]
            y[row] += data[i] * x[col]
            if row != col:
                y[col] += data[i] * x[row]
    else:
        for i in range(self.nnz):
            y[self.row[i]] += data[i] * x[self.col[i]]

@cython.boundscheck(False)
cdef inline matvecCSR(csrdata *self, numeric *x, numeric *y):
    cdef numeric *data = <numeric *>self.data
    cdef numeric tmp
    cdef long i, j, k
    
    if self.isSym:
        for i in range(self.rows):
            tmp = y[i]
            for j in range(self.indptr[i], self.indptr[i + 1]):
                tmp += data[j] * x[self.indices[j]]
            
            for k in range(i + 1, self.rows):
                for j in range(self.indptr[k], self.indptr[k + 1]):
                    if self.indices[j] == i:
                        tmp += data[j] * x[k]
                        break
            y[i] = tmp
    else:
        for i in range(self.rows):
            tmp = y[i]
            for j in range(self.indptr[i], self.indptr[i + 1]):
                tmp += data[j] * x[self.indices[j]]
            y[i] = tmp