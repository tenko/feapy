# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
import sys
import datetime
import math

import numpy as np

from base import FEError, NonStructualElement
from beam import BaseBeam, Beam

class PostProcess(object):
    '''
    Post process functions
    '''
    def fileHeader(self, fh, name = ''):
        date = datetime.datetime.today()
        
        print >>fh, "    1C%s" % name
        print >>fh, "    1UUSER"
        print >>fh, "    1UDATE              %s" % date.strftime("%x")
        print >>fh, "    1UTIME              %s" % date.strftime("%X")
        print >>fh, "    1UHOST"
        print >>fh, "    1UPGM               FelyX"
        print >>fh, "    1UDIR"
        print >>fh, "    1UDBN"
    
    def fileNodes(self, fh):
        # start of node block
        print >>fh, "    2C                                                                   1"
        
        for nid, node in enumerate(self.nodes, start = 1):
            args = nid, node.cx, node.cy, node.cz
            print >>fh, " -1%10d% 12.5E% 12.5E% 12.5E" % args
        
        # end of node block
        print >>fh, " -3"
    
    def fileElements(self, fh):
        # start of element block
        print >>fh, "    3C                                                                   1"
        
        nodemap = {}
        for nid, node in enumerate(self.nodes, start = 1):
            nodemap[node] = nid
            
        for elid, element in enumerate(self.elements, start = 1):
            if isinstance(element, BaseBeam):
                eltype = 11
            else:
                clsname = element.__class__.__name__
                raise FEError("Element %s not supported" % clsname)
            
            args = elid, eltype, 1, 1
            print >>fh, " -1%10d%5d%5d%5d" % args
            
            if eltype == 11:
                n1 = nodemap[element.nodes[0]]
                n2 = nodemap[element.nodes[1]]
                print >>fh, " -2%10d%10d" % (n1,n2)
            
        # end of element block
        print >>fh, " -3"
    
    def fileNodalForces(self, fh):
        # start of node results block
        args = "STEP %5d" % self._filestep
        self._filestep += 1
        print >>fh, "    1P%s" % args
        
        args = self.step, len(self.nodes), 0, self.step
        print >>fh, "  100CL  101% 12.5E%12d                    %2d%5d           1" % args
        print >>fh, " -4  FORC        4    1"
        print >>fh, " -5  F1          1    2    1    0"
        print >>fh, " -5  F2          1    2    2    0"
        print >>fh, " -5  F3          1    2    3    0"
        print >>fh, " -5  ALL         1    2    0    0    1ALL"
        
        forces  = self.nodalForces()
        
        for nid in range(1, len(self.nodes) + 1):
            fx,fy,fz = forces[nid - 1,:]
            args = nid, fx, fy, fz
            print >>fh, " -1%10d% 12.5E% 12.5E% 12.5E" % args
            
        # end of nodal results block
        print >>fh, " -3"
        
    def fileNodalDisp(self, fh):
        # start of node results block
        args = "STEP %5d" % self._filestep
        self._filestep += 1
        print >>fh, "    1P%s" % args
        
        args = self.step, len(self.nodes), 0, self.step
        print >>fh, "  100CL  101% 12.5E%12d                    %2d%5d           1" % args
        print >>fh, " -4  DISP        4    1"
        print >>fh, " -5  D1          1    2    1    0"
        print >>fh, " -5  D2          1    2    2    0"
        print >>fh, " -5  D3          1    2    3    0"
        print >>fh, " -5  ALL         1    2    0    0    1ALL"

        disp = self.nodalDisp(globals = True)
        
        for nid in range(1, len(self.nodes) + 1):
            ux,uy,uz = disp[nid - 1,:]
            args = nid, ux, uy, uz
            print >>fh, " -1%10d% 12.5E% 12.5E% 12.5E" % args
        
        # end of nodal results block
        print >>fh, " -3"
        
    def fileEnd(self, fh):
        print >>fh, " 9999\n"
        
    def printNodalDisp(self, setname = None, globals = False, fh = sys.stdout):
        '''
        Print nodal deformation to casename.dat
        '''
        if setname is None:
            setname = 'NALL'
            nset = self.nall()
        else:
            nset = self.nset[setname]
        
        # nodal displacement
        args = setname, self.step
        msg = " displacements (ux,uy,uz) for set %s and time % 12.5E"
        print >>fh, msg % args
        print >>fh, ""
        
        disp = self.nodalDisp(nset, globals)
        
        sset = sorted(nset)
        rows = len(sset)
        
        for row in range(rows):
            nid = sset[row]
            node = self.nodes[nid]
            
            ux,uy,uz = disp[row,:]
            
            fmt = "%6d % .5E % .5E % .5E"
            args = nid, ux, uy, uz
            print >>fh, fmt % args
        
        print >>fh, ""
        
    def nodalDisp(self, nset = None, globals = False):
        '''
        Nodal deformation over set.
        
        If no nset is given all nodes are selected.
        '''
        if nset is None:
            nset = self.nall()
        
        # return array with sorted nodes
        sset = sorted(nset)
        rows = len(sset)
        
        ret = np.zeros((rows, 3), dtype=float)
        
        for row in range(rows):
            node = self.nodes[sset[row]]
            
            if globals and not node.coordSys is None:
                # local -> global
                T = node.coordSys
                ret[row, :] = np.dot(T.T, node.results[:3])
            else:
                ret[row, :] = node.results[:3]
        
        return ret
        
    def printNodalForces(self, setname = None, totals= 'NO', fh = sys.stdout):
        '''
        Print nodal forces to casename.dat
        '''
        if setname is None:
            setname = 'NALL'
            nset = self.nall()
        else:
            nset = self.nset[setname]
        
        # nodal external forces
        if totals != 'ONLY':
            args = setname, self.step
            msg = " forces (fx,fy,fz) for set %s and time %14.7E"
            print >>fh, msg % args
            print >>fh, ""
        
        forces  = self.nodalForces(nset)
        
        sset = sorted(nset)
        rows = len(sset)

        if totals != 'ONLY':
            for row in range(rows):
                nid = sset[row]
                
                fx,fy,fz = forces[row, :]
                
                fmt = "%6d % .5E % .5E % .5E"
                args = nid, fx, fy, fz
                print >>fh, fmt % args
                
            if totals != 'NO':
                print >>fh, ""
                    
        if totals != 'NO':
            args = setname, self.step
            msg = " total force (fx,fy,fz) for set %s and time %14.7E"
            print >>fh, msg % args
            print >>fh, ""
            
            fx,fy,fz = forces.sum(axis=0)
            
            fmt = "       % .5E % .5E % .5E"
            args = fx, fy, fz
            print >>fh, fmt % args
                
        print >>fh, ""
        
    def nodalForces(self, nset = None):
        '''
        Return nodal forces over set.
        The force includes sum of nodal loads and reactions.
        
        If no nset is given all nodes are selected.
        '''
        if nset is None:
            nset = self.nall()
        
        self.solveReactions(nset)
        
        # return array with sorted nodes
        sset = sorted(nset)
        rows = len(sset)
        
        nodeloads = np.zeros((3,), dtype=float)
        ret = np.zeros((rows, 3), dtype=float)
        
        for row in range(rows):
            nid = sset[row]
            node = self.nodes[nid]
            
            # get node loads from global load vector
            nodeloads[:] = 0.
            for dof in range(3):
                # Get global index of this DOF
                idx = node.indexGM(dof)
                # Check if this DOF is active
                if idx < 0:
                    continue
                
                nodeloads[dof] = self.nodalforces[idx]
            
            # sum of reaction + nodeloads
            ret[row,:] = node.reaction[:3] + nodeloads
        
        return ret
    
    def printElementSectionForces(self, setname = None, dx = 1e9, fh = sys.stdout):
        '''
        Print beam section forces to casename.dat
        '''
        if setname is None:
            setname = 'EALL'
            elset = self.eall()
        else:
            elset = self.elset[setname]
        
        # nodal external forces
        args = setname, self.step
        msg = " section forces (Pos,Nx,Vy,Vz,Txx,Myy,Mzz) for set %s and time %14.7E"
        print >>fh, msg % args
        print >>fh, ""
        
        forces  = self.elementSectionForces(elset, dx)
        
        sset = sorted(elset)
        rows = len(sset)
        
        for row in range(rows):
            elid = sset[row]
            
            fmt = "%6d % .3E % .5E % .5E % .5E % .5E % .5E % .5E"
            for force in forces[row]:
                Pos,Nx,Vy,Vz,Txx,Myy,Mzz = force
                args = elid, Pos, Nx, Vy, Vz, Txx, Myy, Mzz
                print >>fh, fmt % args
        
        print >>fh, ""
        
    def elementSectionForces(self, elset = None, dx = 1e9):
        '''
        Calculate beam section forces
        '''
        if elset is None:
            elset = self.eall()
        
        # return array with sorted elements
        sset = sorted(elset)
        rows = len(sset)
        
        ret = []
        for row in range(rows):
            elid = sset[row]
            
            element = self.elements[elid] 
            res = element.calcSectionForces(dx)
            ret.append(res)
            
        return ret
    
    def elementVolume(self, elset = None):
        '''
        Calculate element volume
        '''
        if elset is None:
            elset = self.eall()
        
        # return array with sorted elements
        sset = sorted(elset)
        rows = len(sset)
        
        res = np.zeros((rows,), dtype=float)
        
        for row in range(rows):
            elid = sset[row]
            element = self.elements[elid] 
            res[row] = element.volume()
        
        return res
    
    def printElementVolume(self, setname = None, totals= 'NO', fh = sys.stdout):
        '''
        Print element volume to casename.dat
        '''
        if setname is None:
            setname = 'EALL'
            elset = self.eall()
        else:
            elset = self.elset[setname]
        
        print >>fh, " volume for set %s" % setname
        print >>fh, ""
        
        volume  = self.elementVolume(elset)
        
        sset = sorted(elset)
        rows = len(sset)

        if totals != 'ONLY':
            for row in range(rows):
                elid = sset[row]
                fmt = "%6d % .5E"
                args = elid, volume[row]
                print >>fh, fmt % args
                
            if totals != 'NO':
                print >>fh, ""
                    
        if totals != 'NO':
            args = setname
            msg = " total volume for set %s"
            print >>fh, msg % args
            print >>fh, ""
            
            fmt = "       % .5E"
            print >>fh, fmt % np.sum(volume)
                
        print >>fh, ""
        
    def printSectionProperties(self):
        filename = self.casename + '.sec'
        fh = open(self.casename + '.sec', 'w')
        
        today = datetime.datetime.today()
        date = today.strftime("%x")
        time = today.strftime("%X")
        
        print >>fh, "**"
        print >>fh, "** %s - meshed section properties" % filename
        print >>fh, "** Date: %s" % date
        print >>fh, "** Time: %s" % time
        print >>fh, "**"
        print >>fh, "**    Ax           Ixx           Iyy           Izz"
        
        args = self.sectionProperties()
        
        print >>fh, "%12.5E, %12.5E, %12.5E, %12.5E" % args
    
    def sectionProperties(self):
        # Calculate geometric properties area, cgy, cgx
        Ax = 0.
        my, mz = 0., 0.
        for element in self.elmap.itervalues():
            if not isinstance(element, NonStructualElement):
                raise FEError('Expected only non structural elements')
            
            area, yc, zc = element.areaprop()
            Ax += area
            my += area * yc
            mz += area * zc
            
        cgy, cgz = my / Ax, mz / Ax
        
        # calculate intertia properties relative to center of profile
        Iyy, Izz = 0., 0.
        for element in self.elmap.itervalues():
            ixy, iyy, izz = element.secprop(cgy,cgz)
            Iyy += iyy
            Izz += izz
        
        return Ax, Iyy + Izz, Iyy, Izz
    
    def printFrequency(self, ev, fmin, fmax, fh = sys.stdout):
        print >>fh, '    E I G E N V A L U E   O U T P U T'
        print >>fh, 'MODE NO    EIGENVALUE                       FREQUENCY'
        print >>fh, '                                    REAL PART          IMAGINARY PART'
        print >>fh, '                          (RAD/TIME)      (CYCLES/TIME)   (RAD/TIME)'

        i = 1
        k = 2.*math.pi
        for val in ev:
            if val < fmin**2 or val > fmax**2:
                continue
            
            if val < 0.:
                args = i, val, 0., 0., math.sqrt(-val)
            else:
                args = i, val, math.sqrt(val), math.sqrt(val)/k, 0.
            
            print >>fh, "%7d  %14.7E  %14.7E  %14.7E  %14.7E" % args
            
            i += 1

        print   >>fh, ""

        