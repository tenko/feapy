/*     CALCULIX - A 3-dimensional finite element program                 */
/*              Copyright (C) 1998 Guido Dhondt                          */
/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as    */
/*     published by the Free Software Foundation; either version 2 of    */
/*     the License, or (at your option) any later version.               */

/*     This program is distributed in the hope that it will be useful,   */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of    */ 
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      */
/*     GNU General Public License for more details.                      */

/*     You should have received a copy of the GNU General Public License */
/*     along with this program; if not, write to the Free Software       */
/*     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.         */

/*
 * seperated from CalculiX.h: otherwise everyone would have to include
 * the spooles header files
 */
#include <misc.h>
#include <FrontMtx.h>
#include <SymbFac.h>
#if USE_MT
#include <MT/spoolesMT.h>
#endif

/* increase this for debugging */
#define DEBUG_LVL	0

struct factorinfo 
{
	int size;
	double cpus[11];
	IV *newToOldIV, *oldToNewIV;
	SolveMap *solvemap;
	FrontMtx *frontmtx;
	SubMtxManager *mtxmanager;
	ETree *frontETree;
	int nthread;
	FILE *msgFile;

};

void *spooles_factor(long *row, long *col, double *data,
                    long neq, long nnz, int symmetryflag);

void spooles_solve(void *ptr, double *b, long neq);

void spooles_cleanup(void *ptr);
