# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
import sys
import os
import re
import math
import StringIO

import numpy as np

from base import FE, NonStructualElement, Element, Node, BoundCon
from base import LineLoad, globalX, globalY, globalZ, TINY
from beam import Beam, Truss, SectionTriangle
from postprocess import PostProcess

class FEError(Exception):
    pass
    
class ParserError(Exception):
    def __init__(self, line, msg):
        self.line = line
        self.msg = msg
    
    def __str__(self):
        args = (self.line.number, self.line, self.msg)
        return "ParserError(LNO=%d\nLINE:'%s'\nMSG:'%s'" % args
    
class Line(str):
    def __new__(self, number, line):
        return str.__new__(self, line)
        
    def __init__(self, number, line):
        self.number = number
        
        if line.upper().startswith('*HEADING'):
            div = ' '
        else:
            div = ','
        
        parts = tuple(part.strip() for part in line.split(div))
        if parts and parts[0] and parts[0][0] == '*':
            self.keyword = parts[0].upper()
            self.parts = parts[1:]
        else:
            self.keyword = ''
            self.parts = parts

class EmptyLine(object):
    keyword = ''
    parts = ()

def isInt(value):
    return bool(re.match(r"^\d+$", value))
            
            
class Parser(FE, PostProcess):
    '''
    Reader of Abaqus like input format
    '''
    int_re = re.compile(r"^(?P<value>[+-]?[\d]+)$")
    
    def __init__(self, filename):
        FE.__init__(self)
        PostProcess.__init__(self)
        
        self.filename = filename
        
        try:
            fh = open(filename)
        except IOError:
            raise FEError("could not open file: '%s'" % filename)
        
        filename = os.path.split(filename)[-1]
        self.casename = os.path.splitext(filename)[0]
        
        self.analysis = None
        self.step = 0
        self.includelevel = 0
        self.matidx = 1
        self.nodemap = {}
        self.elmap = {}
        self.mat = {}
        self.nset = {}
        self.elset = {}
        
        self._postfh = None
        self._printfh = None
        self._filefh = None
        self._filestep = None
        
        self.processFile(fh)
    
    @property
    def printfh(self):
        '''
        Create casename.dat file if requested
        '''
        if self._printfh is None:
            self._printfh = open(self.casename + '.dat', 'w')
        
        return self._printfh
    
    @property
    def filefh(self):
        '''
        Create casename.frd file if requested
        '''
        if self._filefh is None:
            self._filefh = open(self.casename + '.frd', 'w')
            self._filestep = 1
            
            self.fileHeader(self._filefh)
            self.fileNodes(self._filefh)
            self.fileElements(self._filefh)
        
        return self._filefh
        
    def nextline(self, fh, lineno = [0]):
        '''
        Fetch next line in input
        '''
        while True:
            try:
                line = fh.next()
                lineno[0] += 1
            except StopIteration:
                return EmptyLine
            # skip comments
            if len(line) > 1 and line[0:2] == '**':
                continue
            break
        
        return Line(lineno[0], line.rstrip())
            
    def processFile(self, fh):
        # get next line
        line = self.nextline(fh)
        
        # loop over lines
        while not line is EmptyLine:
            if line.keyword == '*HEADING':
                # skip heading
                line = self.nextline(fh)
                while not line.keyword and not line is EmptyLine:
                    line = self.nextline(fh)
            
            elif line.keyword == '*INCLUDE':
                line = self.handleINCLUDE(fh, line)
                
            elif line.keyword == '*NODE':
                line = self.handleNODE(fh, line)
            
            elif line.keyword == '*NSET':
                line = self.handleNSET(fh, line)
                
            elif line.keyword == '*ELEMENT':
                line = self.handleELEMENT(fh, line)
            
            elif line.keyword == '*ELSET':
                line = self.handleELSET(fh, line)
                
            elif line.keyword == '*MATERIAL':
                line = self.handleMATERIAL(fh, line)
            
            elif line.keyword == '*BEAM SECTION':
                line = self.handleBEAMSECTION(fh, line)
            
            elif line.keyword == '*BOUNDARY':
                line = self.handleBOUNDARY(fh, line)
            
            elif line.keyword == '*STEP':
                line = self.handleSTEP(fh, line)
            
            elif line.keyword == '*STATIC':
                self.analysis = 'STATIC'
                line = self.nextline(fh)
            
            elif line.keyword == '*CLOAD':
                line = self.handleCLOAD(fh, line)
            
            elif line.keyword == '*DLOAD':
                line = self.handleDLOAD(fh, line)
                
            elif line.keyword == '*NODE PRINT':
                line = self.handleNODEPRINT(fh, line)
            
            elif line.keyword == '*NODE FILE':
                line = self.handleNODEFILE(fh, line)
            
            elif line.keyword == '*EL PRINT':
                line = self.handleELPRINT(fh, line)
                
            elif line.keyword == '*END STEP':
                line = self.handleENDSTEP(fh, line)
                
            else:
                print >>sys.stderr, "Skipping: '%s'" % line
                line = self.nextline(fh)
    
        if self.includelevel == 0:
            # mark end in .frd file
            if not self._filefh is None:
               self.fileEnd(self._filefh)
               self._filefh.close()
               self._filefh = None
            
            # check for possible section properties calculation
            if self.step == 0:
                section = False
                for element in self.elmap.itervalues():
                    if isinstance(element, NonStructualElement):
                        section = True
                        break
                
                if section:
                    self.handleSection()
                    
        else:
            self.includelevel -= 1
    
    def handleSection(self):
        self.printSectionProperties()
        
    def handleINCLUDE(self, fh, line):
        if self.includelevel > 10:
            raise FEError("*INCLUDE level > 10")
        
        self.includelevel += 1
        
        filename = None
        for part in line.parts:
            if part.upper().startswith("INPUT="):
                _, filename = part.split("=")
                
            else:
                raise ParserError(line, '*INCLUDE definition malformed!')
        
        if filename is None:
            raise ParserError(line, '*INCLUDE definition malformed!')
        
        try:
            fhi = open(filename)
        except IOError:
            raise FEError("could not open file: '%s'" % filename)
        
        self.processFile(fhi)
        
        return self.nextline(fh)
        
    def handleNODE(self, fh, line):
        if self.step > 0:
            raise FEError('*NODE should be placed before all step definitions')
        
        nset = None
        nsetdat = set()
        
        # check for nset definition
        for part in line.parts:
            part = part.upper()
            if part.startswith("NSET="):
                _, nset = part.split("=")
            else:
                raise ParserError(line, '*NODE definition malformed!')
            
        # loop until EOF or new keyword
        line = self.nextline(fh)
        while not line.keyword and not line is EmptyLine:
            nid = self.parse_int(line.parts[0], minimum = 1)
            if nid in self.nodemap:
                raise FEError("Nodes #%d redefined" % nid)
            
            # add to set
            if not nset is None:
                nsetdat.add(nid)
            
            # fetch coordinates
            x, y, z = 0., 0., 0.
            coords = line.parts[1:]
            
            ll = len(coords)
            if ll == 1:
                x = float(coords[0])
            elif ll == 2:
                x, y = map(float, coords)
            elif ll == 3:
                x, y, z = map(float, coords)
            else:
                raise ParserError(line, '*NODE definition malformed!')
            
            self.nodemap[nid] = Node(x, y, z)
            
            # get next line
            line = self.nextline(fh)
        
        # Save or update nset
        if not nset is None:
            if nset in self.nset:
                self.nset[nset] += nsetdat
            else:
                self.nset[nset] = nsetdat
        
        return line
    
    def handleNSET(self, fh, line):
        if self.step > 0:
            raise FEError('*NSET should be placed before all step definitions')
        
        nset = None
        nsetdat = set()
        generate = False
        
        # check for nset definition
        for part in line.parts:
            part = part.upper()
            if part.startswith("NSET="):
                _, nset = part.split("=")
            
            elif part == "GENERATE":
                generate = True
                
            else:
                raise ParserError(line, '*NSET definition malformed!')
        
        if nset is None:
            raise ParserError(line, 'Expected NSET name')
            
        # loop until EOF or new keyword
        line = self.nextline(fh)
        while not line.keyword and not line is EmptyLine:
            if generate:
                step = 1
                ll = len(line.parts)
                if ll == 3:
                    start, stop, step = line.parts
                    step = self.parse_int(step, minimum = 1)
                elif ll == 2:
                    start, stop = line.parts
                else:
                    raise ParserError(line, '*NSET definition malformed!')
                
                start = self.parse_int(start, minimum = 1)
                stop = self.parse_int(stop, minimum = 1)
                
                nsetdat.update(range(start, stop + 1, step))
            else:
                if len(line.parts) > 16:
                    raise ParserError(line, '*NSET definition to long')
                
                nsetdat.update(self.parse_int(val, minimum = 1) for val in line.parts)
                
            # get next line
            line = self.nextline(fh)
        
        # Save or update nset
        if nset in self.nset:
            self.nset[nset] += nsetdat
        else:
            self.nset[nset] = nsetdat
        
        return line
        
    def handleELEMENT(self, fh, line):
        if self.step > 0:
            raise FEError('*ELEMENT should be placed before all step definitions')
        
        elset = None
        elsetdat = set()
        
        eltype = None
        for part in line.parts:
            part = part.upper()
            if part.startswith("ELSET="):
                _, elset = part.split("=")
            elif part.startswith("TYPE="):
                _, eltype = part.split("=")
            else:
                raise ParserError(line, '*ELEMENT definition malformed!')
        
        if eltype is None:
            raise ParserError(line, '*ELEMENT type not defined!')
        elif eltype == 'B31':
            cls, nnodes = Beam, 2
        elif eltype == 'T3D2':
            cls, nnodes = Truss, 2
        elif eltype == 'S3':
            cls, nnodes = SectionTriangle, 3
        else:
            raise ParserError(line, 'Unknown element type "%s"' % eltype)
        
        # loop until EOF or new keyword
        line = self.nextline(fh)
        while not line.keyword and not line is EmptyLine:
            elid = int(line.parts[0])
            if elid in self.elmap:
                raise FEError("Element #%d redefined" % elid)
            
            # add to set
            if not elset is None:
                elsetdat.add(elid)
            
            # fetch nodes
            nodes = [self.nodemap[int(nid)] for nid in line.parts[1:]]
            if len(nodes) != nnodes:
                raise ParserError(line, 'Element has wrong number of nodes')
            
            self.elmap[elid] = cls(*nodes)
            
            # get next line
            line = self.nextline(fh)
        
        # Save or update elset
        if not elset is None:
            if elset in self.elset:
                self.elset[elset].update(elsetdat)
            else:
                self.elset[elset] = elsetdat
        
        return line

    def handleELSET(self, fh, line):
        if self.step > 0:
            raise FEError('*ELSET should be placed before all step definitions')
        
        elset = None
        elsetdat = set()
        generate = False
        
        # check for elset definition
        for part in line.parts:
            part = part.upper()
            if part.startswith("ELSET="):
                _, elset = part.split("=")
            
            elif part == "GENERATE":
                generate = True
                
            else:
                raise ParserError(line, '*ELSET definition malformed!')
        
        if elset is None:
            raise ParserError(line, 'Expected ELSET name')
            
        # loop until EOF or new keyword
        line = self.nextline(fh)
        while not line.keyword and not line is EmptyLine:
            if generate:
                step = 1
                ll = len(line.parts)
                if ll == 3:
                    start, stop, step = line.parts
                    step = self.parse_int(step, minimum = 1)
                elif ll == 2:
                    start, stop = line.parts
                else:
                    raise ParserError(line, '*ELSET definition malformed!')
                
                start = self.parse_int(start, minimum = 1)
                stop = self.parse_int(stop, minimum = 1)
                
                elsetdat.update(range(start, stop + 1, step))
            else:
                if len(line.parts) > 16:
                    raise ParserError(line, '*ELSET definition to long')
                
                elsetdat.update(self.parse_int(val, minimum = 1) for val in line.parts)
                
            # get next line
            line = self.nextline(fh)
        
        # Save or update elset
        if elset in self.elset:
            self.elset[elset] += elsetdat
        else:
            self.elset[elset] = elsetdat
        
        return line
        
    def handleMATERIAL(self, fh, line):
        if self.step > 0:
            raise FEError('*MATERIAL should be placed before all step definitions')
        
        mat = {}
        
        # check for name
        name = None
        for part in line.parts:
            part = part.upper()
            if part.startswith("NAME="):
                _, name = part.split("=")
            else:
                raise ParserError(line, '*MATERIAL definition malformed!')
        
        if name is None:
            raise ParserError(line, 'Expected *MATERIAL name!')
        
        # fetch material definition
        line = self.nextline(fh)
        if not line.keyword or line is EmptyLine:
            raise ParserError(line, 'Expected *MATERIAL definitions!')
        
        while True:
            if line.keyword == '*ELASTIC':
                if 'class' in mat:
                    raise ParserError(line, '*ELASTIC already defined!')
                else:
                    mat['class'] = 'ELASTIC'
                
                for part in line.parts:
                    part = part.upper()
                    if part.startswith("TYPE="):
                        _, name = part.split("=")
                        if name != 'ISO':
                            raise FEError("Only ISO ELASTIC material supported")
                    else:
                        raise ParserError(line, '*ELASTIC definition malformed!')
                
                line = self.nextline(fh)
                if line.keyword or line is EmptyLine:
                    raise ParserError(line, 'Expected *ELASTIC definitions!')
                
                ll = len(line.parts)
                if ll == 3:
                    E, nu, T = line.parts
                elif ll == 2:
                    E, nu = line.parts
                else:
                    raise ParserError(line, '*ELASTIC definition malformed!')
                
                mat['E'] = self.parse_float(E, minimum = TINY)
                mat['nu'] = self.parse_float(nu, minimum = TINY, maximum = 1.)
                
            elif line.keyword == '*DENSITY':
                if 'density' in mat:
                    raise ParserError(line, '*DENSITY already defined!')
                
                line = self.nextline(fh)
                if line.keyword or line is EmptyLine:
                    raise ParserError(line, 'Expected *DENSITY definitions!')
                
                ll = len(line.parts)
                if ll == 1 or ll == 2:
                    mat['density'] = self.parse_float(line.parts[0], minimum = TINY)
                else:
                    raise ParserError(line, '*DENSITY definition malformed!')
            
            else:
                break
                
            # fetch next line
            line = self.nextline(fh)
            
            if not 'class' in mat:
                raise ParserError(line, '*MATERIAL definitions missing!')
            
            self.mat[name] = mat
                      
        return line

    def handleBEAMSECTION(self, fh, line):
        mat = {}
        prop = {}
        
        elset = None
        section = 'GENERAL'
        
        for part in line.parts:
            part = part.upper()
            if part.startswith("ELSET="):
                _, elset = part.split("=")
                
            elif part.startswith("SECTION="):
                _, section = part.split("=")
                
            else:
                raise ParserError(line, '*BEAM SECTION definition malformed!')
        
        # fetch next line
        line = self.nextline(fh)
        if line.keyword or line is EmptyLine:
            raise ParserError(line, 'Expected *BEAM SECTION definitions!')
        
        if section == 'GENERAL':
            step = 0
            while not line.keyword and not line is EmptyLine:
                if step == 0:
                    if len(line.parts) == 6:
                        prop['ShearY'] = self.parse_float(line.parts[4], minimum = 0.)
                        prop['ShearZ'] = self.parse_float(line.parts[5], minimum = 0.)
                    elif len(line.parts) == 4:
                        prop['ShearY'] = 0.
                        prop['ShearZ'] = 0.
                    else:
                        raise ParserError(line, '*BEAM SECTION definition malformed!')
                    
                    prop['Area'] = self.parse_float(line.parts[0], minimum = TINY)
                    prop['Ixx'] = self.parse_float(line.parts[1], minimum = TINY)
                    prop['Iyy'] = self.parse_float(line.parts[2], minimum = TINY)
                    prop['Izz'] = self.parse_float(line.parts[3], minimum = TINY)
                
                elif step == 1 and line:
                    nx, ny, nz = line.parts
                    nx = self.parse_float(nx, minimum = 0., maximum = 1.)
                    ny = self.parse_float(ny, minimum = 0., maximum = 1.)
                    nz = self.parse_float(nz, minimum = 0., maximum = 1.)
                    
                    # unitize vector
                    n = np.array((nx,ny,nz))
                    n = n / np.linalg.norm(n)
                    
                    # project to yz plane
                    t = np.array((1.,0.,0.))
                    n = n - np.dot(n,t)*t
                    n = n / np.linalg.norm(n)
                    
                    # angle between normal and section dir 2
                    zaxis = np.array((0., 0., 1.))
                    theta = math.acos(np.dot(n, zaxis))
                    if abs(theta) > TINY:
                        prop['Theta'] = theta
                    
                elif step == 2:
                    # material properties
                    mat['E'] = self.parse_float(line.parts[0], minimum = TINY)
                    mat['G'] = self.parse_float(line.parts[1], minimum = TINY)
                    mat['nu'] = self.parse_float(line.parts[2], minimum = TINY, maximum = 1.)
                    
                    # optional density
                    if len(line.parts) == 4:
                        mat['density'] = self.parse_float(line.parts[3], minimum = TINY)
                
                else:
                    raise ParserError(line, '*BEAM SECTION definition malformed!')
                
                step += 1
                
                # fetch next line
                line = self.nextline(fh)
        else:
            raise ParserError(line, "Section type '%s' not supported!" % section)
        
        # apply to selected set
        selection = self.elset[elset]
        for elid in selection:
            element = self.elmap[elid]
            element.material = mat
            element.properties.update(prop)
        
        return line

    def handleBOUNDARY(self, fh, line):
        for part in line.parts:
            part = part.upper()
            if part.startswith("OP="):
                _, op = part.split("=")
                # remove all previous defined boundaries
                if op == 'NEW':
                    if self.step <= 1:
                        nodes = self.nodemap.itervalues()
                    else:
                        nodes = self.nodes
                        
                    for node in nodes:
                        if not node.boundCon is None:
                            node.boundCon.deleteBounds()
            else:
                raise ParserError(line, '*BOUNDARY definition malformed!')
        
        # fetch next line
        line = self.nextline(fh)
        if line.keyword or line is EmptyLine:
            raise ParserError(line, 'Expected *BOUNDARY definitions!')
        
        if self.step <= 1:
            nodes = self.nodemap
        else:
            nodes = self.nodes
                        
        def applyBoundary(selection, start, stop = None, value = None):
            if isInt(selection):
                selection = (int(selection),)
            else:
                selection = self.nset[selection.upper()]
            
            start = self.parse_int(start, minimum = 1, maximum = 6)
            
            if stop is None:
                stop = start + 1
            else:
                stop = self.parse_int(stop, minimum = 1, maximum = 6) + 1
                
            if value is None:
                if self.step != 0:
                    raise FEError('Homogeneous conditions inside *STEP')
                value = 0.
            else:
                value = self.parse_float(value)
                
            for nid in selection:
                node = nodes[nid]
                
                if node.boundCon is None:
                    node.boundCon = BoundCon()
                
                for dof in range(start, stop):
                    node.boundCon[dof - 1] = 0.
        
        while not line.keyword and not line is EmptyLine:
            if len(line.parts) in (2, 3, 4):
                applyBoundary(*line.parts)
            else:
                raise ParserError(line, '*BOUNDARY definition malformed!')
            
            line = self.nextline(fh)
        
        return line

    def handleSTEP(self, fh, line):
        self.analysis = None
        self.step += 1
        self._postfh = StringIO.StringIO()
        
        return self.nextline(fh)

    def handleDLOAD(self, fh, line):
        for part in line.parts:
            part = part.upper()
            if part.startswith("OP="):
                _, op = part.split("=")
                # remove all previous defined boundaries
                if op == 'NEW':
                    if self.step <= 1:
                        elements = self.elmap.itervalues()
                    else:
                        elements = self.elements
                        
                    for element in elements:
                        del element.loads[:]
                        
            else:
                raise ParserError(line, '*CLOAD definition malformed!')
        
        # fetch next line
        line = self.nextline(fh)
        if line.keyword or line is EmptyLine:
            raise ParserError(line, 'Expected *DLOAD definitions!')

        if self.step <= 1:
            elements = self.elmap
        else:
            elements = self.elements
                        
        while not line.keyword and not line is EmptyLine:
            if len(line.parts) < 3:
                raise ParserError(line, '*DLOAD definition malformed!')
            
            selection = line.parts[0]
            if isInt(selection):
                selection = (int(selection),)
            else:
                selection = self.elset[selection.upper()]
            
            loadtype = line.parts[1]
            if loadtype == 'GRAV':
                mag = self.parse_float(line.parts[2], minimum = TINY)
                
                nx = self.parse_float(line.parts[3], minimum = -1., maximum = 1.)
                ny = self.parse_float(line.parts[4], minimum = -1., maximum = 1.)
                nz = self.parse_float(line.parts[5], minimum = -1., maximum = 1.)
                
                for elid in selection:
                    element = elements[elid]
                    element.applyGravity(mag, nx, ny, nz)
            
            elif loadtype in ('PX', 'PY', 'PZ', 'P2', 'P3'):
                mag = self.parse_float(line.parts[2])
                
                for elid in selection:
                    element = elements[elid]
                    if isinstance(element, Beam):
                        element.applyLineLoad(mag, loadtype)
                    else:
                        element.applyPressure(mag, loadtype)
                
            else:
                raise ParserError(line, '*DLOAD definition malformed!')
                    
            line = self.nextline(fh)
        
        return line
        
    def handleCLOAD(self, fh, line):
        for part in line.parts:
            part = part.upper()
            if part.startswith("OP="):
                _, op = part.split("=")
                # remove all previous defined boundaries
                if op == 'NEW':
                    if self.step == 1:
                        nodes = self.nodemap.itervalues()
                    else:
                        nodes = self.nodes
                        
                    for node in nodes:
                        if not node.boundCon is None:
                            node.boundCon.deleteLoads()
            else:
                raise ParserError(line, '*CLOAD definition malformed!')
        
        # fetch next line
        line = self.nextline(fh)
        if line.keyword or line is EmptyLine:
            raise ParserError(line, 'Expected *CLOAD definitions!')
        
        while not line.keyword and not line is EmptyLine:
            if len(line.parts) != 3:
                raise ParserError(line, '*CLOAD definition malformed!')
            
            selection, dof, value = line.parts
            
            if isInt(selection):
                selection = (int(selection),)
            else:
                selection = self.nset[selection.upper()]
            
            dof = self.parse_int(dof, minimum = 1, maximum = 6)
            value = self.parse_float(value)
            
            for nid in selection:
                node = self.nodemap[nid]
                
                if node.boundCon is None:
                    node.boundCon = BoundCon()
                
                node.boundCon[dof + 5] = value
                    
            line = self.nextline(fh)
        
        return line

    def handleNODEPRINT(self, fh, line):
        if self.analysis != 'POST':
            self._postfh.write(line)
            self._postfh.write('\n')
            
            # fetch next line
            line = self.nextline(fh)
            if line.keyword or line is EmptyLine:
                msg = 'Expected *NODE PRINT definitions!'
                raise ParserError(line, msg)
            
            self._postfh.write(line)
            self._postfh.write('\n')
            
        else:
            nset = None
            totals = 'NO'
            dat = {}
            
            for part in line.parts:
                part = part.upper()
                if part.startswith("NSET="):
                    _, nset = part.split("=")
                    
                elif part.startswith("TOTALS="):
                    _, totals = part.split("=")
                    
                else:
                    raise ParserError(line, '*NODE PRINT definition malformed!')
            
            if nset is None or nset not in self.nset:
                raise FEError("NSET not valid")
                
            # fetch next line
            line = self.nextline(fh)
            
            pfh = self.printfh
            for arg in line.parts:
                if arg == 'U':
                    self.printNodalDisp(nset, self.printfh)

                elif arg == 'RF':
                    self.printNodalForces(nset, totals, self.printfh)
                
                else:
                    raise FEError("Key '%s' not known" % arg)
                    
        return self.nextline(fh)

    def handleNODEFILE(self, fh, line):
        if self.analysis != 'POST':
            if line.parts:
                msg = 'No keywords expected'
                raise ParserError(line, msg)
            
            self._postfh.write(line)
            self._postfh.write('\n')
            
            # fetch next line
            line = self.nextline(fh)
            if line.keyword or line is EmptyLine:
                msg = 'Expected *NODE FILE definitions!'
                raise ParserError(line, msg)
            
            self._postfh.write(line)
            self._postfh.write('\n')
            
        else:
            # fetch next line
            line = self.nextline(fh)
            
            for arg in line.parts:
                if arg == 'U':
                    self.fileNodalDisp(self.filefh)

                elif arg == 'RF':
                    self.fileNodalForces(self.filefh)
                
                else:
                    raise FEError("Key '%s' not known" % arg)
                    
        return self.nextline(fh)
    
    def handleELPRINT(self, fh, line):
        if self.analysis != 'POST':
            self._postfh.write(line)
            self._postfh.write('\n')
            
            # fetch next line
            line = self.nextline(fh)
            if line.keyword or line is EmptyLine:
                msg = 'Expected *EL PRINT definitions!'
                raise ParserError(line, msg)
            
            self._postfh.write(line)
            self._postfh.write('\n')
            
        else:
            elset = None
            totals = 'NO'
            dx = 1e9
            
            for part in line.parts:
                part = part.upper()
                if part.startswith("ELSET="):
                    _, elset = part.split("=")
                
                elif part.startswith("TOTALS="):
                    _, totals = part.split("=")
                    
                elif part.startswith("SECTIONDELTA="):
                    _, dx = part.split("=")
                    dx = self.parse_float(dx, minimum = 0.)
                    
                else:
                    raise ParserError(line, '*EL PRINT definition malformed!')
            
            if elset is None or elset not in self.elset:
                raise FEError("ELSET not given")
                
            # fetch next line
            line = self.nextline(fh)
            
            pfh = self.printfh
            for arg in line.parts:
                if arg == 'SF':
                    self.printElementSectionForces(elset, dx, self.printfh)
                
                elif arg == 'EVOL':
                    self.printElementVolume(elset, totals, self.printfh)
                    
                else:
                    raise FEError("Key '%s' not known" % arg)
                    
        return self.nextline(fh)
        
    def handleENDSTEP(self, fh, line):
        if self.analysis != 'STATIC':
            raise FEError("Analysis '%s' not supported" % self.analysis)
            
        if self.step == 1:
            print >>sys.stdout, "Model size:\n"
            
            # store node in sequence
            for nid in range(1, len(self.nodemap) + 1):
                if not nid in self.nodemap:
                    raise FEError('Node #%d missing' % nid)
                
                node = self.nodemap[nid]
                self.nodes.append(node)
            
            self.nodemap = None
            print >>sys.stdout, "   nodes:    %d" % len(self.nodes)
            
            # store elements in sequence
            for elid in range(1, len(self.elmap) + 1):
                if not elid in self.elmap:
                    raise FEError('Element #%d missing' % elid)
                
                element = self.elmap[elid]
                if isinstance(element, NonStructualElement):
                    raise FEError("Non structural element")
                self.elements.append(element)
            
            self.elmap = Node
            print >>sys.stdout, "   elements: %d" % len(self.elements)
            print >>sys.stdout, ""
            
            # sanity check of model
            self.validate()
            
            # Link node DOF's to GSM index
            self.linkNodes()
            
            # Eval envelope and profile of GSM
            self.evalEnvelope()
        
        print >>sys.stdout, "STEP:        %d" % self.step
        print >>sys.stdout, "Analysis:    %s" % self.analysis
        print >>sys.stdout, ""
        print >>sys.stdout, "Number of equations:     %d" % self.dofcount
        print >>sys.stdout, "Nonzero matrix elements: %d" % np.sum(self.envelope)
        print >>sys.stdout, ""
        
        # Evaluate ESM's and assemble them to GSM
        self.assembleElementK()
        
        # Apply nodal loads
        self.applyLoads()
        
        # solve
        self.directSolver()
        
        # store deformation in nodes
        self.storeNodalResults()
        
        # post process results
        if self._postfh.getvalue():
            self.analysis = 'POST'
            self._postfh.seek(0)
            self.processFile(self._postfh)
        
        return self.nextline(fh)
    
    @classmethod
    def parse_float(cls, value, minimum = None, maximum = None, line = EmptyLine):
        try:
            value = float(value)
        except ValueError:
            raise ParserError(line, "'%s' not a valid float" % value)
            
        if not minimum is None:
            if value < minimum:
                raise ParserError(line, "'%g' smaller than minimum" % value)
        
        if not maximum is None:
            if value > maximum:
                raise ParserError(line, "'%g' larger than maximum" % value)
        
        return value
    
    @classmethod
    def parse_int(cls, value, minimum = None, maximum = None, line = EmptyLine):
        match = cls.int_re.match(value)
        if match is None:
            raise ParserError(line, "'%s' not a valid int" % value)
            
        value = int(value)
            
        if not minimum is None:
            if value < minimum:
                raise ParserError(line, "'%d' smaller than minimum" % value)
        
        if not maximum is None:
            if value > maximum:
                raise ParserError(line, "'%d' larger than maximum" % value)
        
        return value
        
if __name__ == '__main__':
    fe = Parser('frame_rotated_section.inp')
    #os.chdir('tests')
    #fe = Parser('frame_tower.inp')
    
    #fe.printNodalDisp()
    #fe.printNodalForces(totals='YES')
    #fe.printElementSectionForces()