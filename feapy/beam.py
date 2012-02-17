# -*- coding: utf-8 -*-
#
# This file is part of feapy - See LICENSE.txt
#
from math import pi, sin, cos, acos, atan2
from math import sqrt, pow, hypot
from math import radians, degrees 
        
import numpy as np

from base import List, DofSet, Properties, LineLoad
from base import globalX, globalY, globalZ
from base import localX, localY, localZ
from base import Element, NonStructualElement, Material
from base import TINY, FEError
    
class SectionTriangle(NonStructualElement):
    '''
    Non structural 2D triangle element to
    define beam cross sections
    '''
    def __init__(self, n1, n2, n3, material = None, properties = None):
        BaseBeam.__init__(self)
        
        self.nodeCount = 3
        self.name = 'Non Structural 2D Triangle'
        
        self.nodes = (n1,n2,n3)
        self.material = material
        self.loads = List()
        
        if properties is None:
            properties = Properties()
        self.properties = properties
    
    def __str__(self):
        return '()'
    
    def __repr__(self):
        return '%s%s' % (self.__class__.__name__, str(self))
    
    def areaprop(self):
        '''
        Calculate area and center of triangle
        '''
        x0 = self.nodes[0].cx
        y0 = self.nodes[0].cy
        
        x1 = self.nodes[1].cx
        y1 = self.nodes[1].cy
        
        x2 = self.nodes[2].cx
        y2 = self.nodes[2].cy
        
        # Triangle center and area
        xc = (x0 + x1 + x2)/3.
        yc = (y0 + y1 + y2)/3.
        area = abs(.5*(x0*y1 + x1*y2 + x2*y0 - x1*y0 - x2*y1 - x0*y2))
        
        return area, xc, yc
    
    def secprop(self, cgy, cgz):
        '''
        Calculate section properties relative to given
        center cgx,cgy
        '''
        x0 = self.nodes[0].cx
        y0 = self.nodes[0].cy
        
        x1 = self.nodes[1].cx
        y1 = self.nodes[1].cy
        
        x2 = self.nodes[2].cx
        y2 = self.nodes[2].cy
        
        # Triangle center and area
        _ycg = (x0 + x1 + x2)/3.
        _zcg = (y0 + y1 + y2)/3.
        _area = abs(.5*(x0*y1 + x1*y2 + x2*y0 - x1*y0 - x2*y1 - x0*y2))
        
        b = hypot(x1 - x0, y1 - y0)
        a = ((x2 - x0) * (x1 - x0) + (y2 - y0) * (y1 - y0)) / b
        h = sqrt(abs(pow(hypot(x2 - x1, y2 - y1), 2.) - pow(b - a, 2.)))
        
        # local to triangle
        _Iyy = b*pow(h, 3.) / 36.
        _Izz = (h*pow(b, 3.) - pow(b, 2.)*h*a + b*h*pow(a,2.)) / 36.
        _Iyz = (pow(b, 2.)*pow(h, 2.) - 2.*b*pow(h, 2.)*a) / 72.
        
        # Rotated axis and realtive to cgx and cgy
        angle = 2.*atan2(y1 - y0, x1 - x0)
        
        Iyy = .5*(_Iyy + _Izz) + .5*(_Iyy - _Izz) * cos(angle) - _Iyz * sin(angle) \
              + _area * pow(cgz - _zcg, 2.)
              
        Izz = .5*(_Iyy + _Izz) - .5*(_Iyy - _Izz) * cos(angle) + _Iyz * sin(angle) \
              + _area * pow(cgy - _ycg, 2.)
              
        Iyz = .5*(_Iyy - _Izz) * sin(angle) + _Iyz * cos(angle) \
              + _area * (_ycg - cgy) * (_zcg - cgz)
        
        return Iyz, Iyy, Izz

        
class BaseBeam(Element):
    def length(self):
        '''
        Eval length of element
        '''
        n1 = self.nodes[0]
        n2 = self.nodes[1]
        
        return sqrt((n1.cx - n2.cx)**2 + (n1.cy - n2.cy)**2 + \
                     (n1.cz - n2.cz)**2)
    
    def volume(self):
        '''
        Evaluate volume of the element
        '''
        return self.length() * self.properties['Area']
    
    def applyLineLoad(self, force, loadtype):
        '''
        Apply line load.
        
        force = load per length unit
        loadtype = PX', PY', PZ', P2', P3
        
        PX : global x direction
        PY : global y direction
        PZ : global  direction
        P2 : beam local direction 2 (beam y axis)
        P3 : beam local direction 3 (beam z axis)
        '''
        if loadtype == 'PX':
            direction = globalX
        elif loadtype == 'PY':
            direction = globalY
        elif loadtype == 'PZ':
            direction = globalZ
        elif loadtype == 'P2':
            direction = localY
        elif loadtype == 'P3':
            direction = localZ
        else:
            raise FEError("loadtype '%s' not supported" % loadtype)
        
        self.loads.append(LineLoad(force, direction))
        
    def applyGravity(self, magnitude, nx, ny, nz):
        '''
        Apply gravitational load.
        
        magnitude = acceleration of gravity
        nx, ny, nz = gravity vector direction
        '''
        
        if 'density' not in self.material:
            raise FEError('Material density not defined')
            
        density = self.material['density']
        mass = self.volume()*density
        weight = magnitude*mass/self.length()
        
        if nx != 0.:
            load = LineLoad(nx*weight, globalX)
            self.loads.append(load)
        
        if ny != 0.:
            load = LineLoad(ny*weight, globalY)
            self.loads.append(load)
        
        if nz != 0.:
            load = LineLoad(nz*weight, globalZ)
            self.loads.append(load)
    
    def calcLocalLoad(self):
        '''
        Calculate resultant of line loads
        in local coordinates.
        '''
        Tl = self.calcT()
            
        # beam direction cos
        nx = np.dot(Tl, (1.,0.,0.))
        ny = np.dot(Tl, (0.,1.,0.))
        nz = np.dot(Tl, (0.,0.,1.))
        
        force = res = np.zeros((3,), dtype = float)
        
        for load in self.loads:
            if load.direction == globalX:
                dx, dy, dz = nx
            elif load.direction == globalY:
                dx, dy, dz = ny
            elif load.direction == globalZ:
                dx, dy, dz = nz
            elif load.direction == localX:
                dx, dy, dz = 1., 0., 0.
            elif load.direction == localY:
                dx, dy, dz = 0., 1., 0.
            else:
               dx, dy, dz = 0., 0., 1.
            
            force[0] += load.force*dx
            force[1] += load.force*dy
            force[2] += load.force*dz
        
        return force
        
    def calcT(self):
        '''
        Eval 3x3 transformation matrix from global to local coordinates.
        '''
        T = np.zeros((3,3), dtype=float)
        
        n1 = self.nodes[0]
        n2 = self.nodes[1]
        
        L = self.length()
        
        # Check if element length axis is within a 0.01 percent slope of global z-axis
        # using direction cosine cxz
        if abs((n2.cz - n1.cz) / L) >= 1/1.000000005:
            # 2a) If true, element y-axis is equal global y-axis 
            # mapping local x to global z
            # local y to global y
            # local z to -global x
            T[0,2] = 1.
            T[1,1] = 1.
            T[2,0] = -1.
            
            # Check in which direction on z-axis the element is oriented -> switch signs
            # accordingly ( THIS EFFECT WAS FOUND AFTER 4 DAYS OF DEBUGGING!!!)
            if n1.cz > n2.cz:
                T[0,2] = -1.
                T[2,0] = 1.
        else:
            # 2b) If false, element y-axis is defined to be in global  xy-plane
            # local x is defined through the nodes
            # mapping to global coords through direction cosines...
            T[0,0] = (n2.cx - n1.cx) / L
            T[0,1] = (n2.cy - n1.cy) / L
            T[0,2] = (n2.cz - n1.cz) / L
            
            # Using that the projection of x to global xy-plane is perpendicular to local y
            Lx = sqrt(T[0,0]**2 + T[0,1]**2)
            T[1,0] = -T[0,1] / Lx
            T[1,1] = T[0,0] / Lx
            T[1,2] = 0.
            
            # direction cosines of z direction are evaluated using cross product
            # of x and y vectors (in "direction cosine coordinates")
            T[2,0] = T[0,1] * T[1,2] - T[0,2] * T[1,1]
            T[2,1] = -( T[0,0] * T[1,2] - T[0,2] * T[1,0] )
            T[2,2] = T[0,0] * T[1,1] - T[0,1] * T[1,0]
    
        # If theta != 0: 
        # From general orientation defined through two nodes and Theta to 
        # orientation with x in direction of nodes, y in global xy-plane
        if self.properties.get('Theta', 0.) != 0.:
            T1 = np.zeros((3,3), dtype=float)
            
            theta = self.properties['Theta']
            co = cos(theta)
            si = sin(theta)
            
            T1[0,0] = 1.
            T1[1,1] = co ; T1[1,2] = si
            T1[2,1] = -si; T1[2,2] = co
            
            T = np.dot(T1,T)
        
        return T
        
class Truss(BaseBeam):
    '''
    2-Node Structural 3D Truss
    '''
    def __init__(self, n1, n2, material = None, properties = None):
        BaseBeam.__init__(self)
        
        self.nodeCount = 2
        self.name = 'Structural 3D Truss'
        
        self.dofSet = DofSet(Dx=True, Dy=True, Dz=True)
        
        self.nodes = (n1,n2)
        self.material = material
        self.loads = List()
        
        if properties is None:
            properties = Properties()
        self.properties = properties
    
    def __str__(self):
        return '()'
    
    def __repr__(self):
        return '%s%s' % (self.__class__.__name__, str(self))
    
    def validate(self):
        '''
        Validate truss
        '''
        if self.length() < TINY:
            raise FEError('Truss length is to small')
        
        # check material
        for name in ('E',):
            if name not in self.material:
                raise FEError('Material definition not complete')
            
            if self.material[name] < TINY:
                raise FEError("Material parameter '%s' not correct" % name)
        
        # check profile data
        for name in ('Area',):
            if name not in self.properties:
                raise FEError('Properties definition not complete')
            
            if self.properties[name] < TINY:
                raise FEError("Properties parameter '%s' not correct" % name)
    
    def calcLocalNodalForces(self, res = None):
        '''
        Calculate corresponding end forces at nodes from
        line loads in local directions
        '''
        if res is None:
            res = np.zeros((self.dofSet.count() * self.nodeCount,), dtype = float)
        
        load = self.calcLocalLoad()
        
        # Nx
        res[0] += .5*load[0]
        res[3] += .5*load[0]
        # Vy
        res[1] += .5*load[1]
        res[4] += .5*load[1]
        # VZ
        res[2] += .5*load[2]
        res[5] += .5*load[2]
        
        return res
        
    def calcNodalForces(self, res = None):
        '''
        Calculate corresponding global forces at nodes from
        line loads
        '''
        # calculate local forces
        res = self.calcLocalNodalForces(res)
        
        # Transforamtion matrix
        T = self.calcT()
        
        res[0:3] = np.dot(T.T, res[0:3])
        res[3:6] = np.dot(T.T, res[3:6])
        
        return res
    
    def calcKe(self):
        '''
        Eval element stiffness matrix in local coordinates
        '''
        mat = self.material
        prop = self.properties
        
        L = self.length()
        Ax, E = prop["Area"], mat["E"]
        
        # Stiffness matrix
        Ke = np.zeros((2,2), dtype=float)
        
        AxEoverL = Ax*E/L
        Ke[0,0] = AxEoverL; Ke[0,1] = -AxEoverL
        Ke[1,0] =-AxEoverL; Ke[1,1] =  AxEoverL
        
        return Ke
        
    def calcK(self):
        '''
        Eval element stiffness matrix in global coordinates.
        '''
        L = self.length()
        n1, n2 = self.nodes
        
        # Transforamtion matrix
        T = np.zeros((2,6), dtype=float)
        
        # Eval direction cosines in order to transform stiffness matrix
        # from local coords (link direction) to global coordinates
        cx = ( n2.cx - n1.cx ) / L
        cy = ( n2.cy - n1.cy ) / L
        cz = ( n2.cz - n1.cz ) / L
        
        T[0,0] = cx; T[0,1] = cy; T[0,2] = cz
        T[1,3] = cx; T[1,4] = cy; T[1,5] = cz
        
        # Stiffness matrix
        K = self.calcKe()
        
        # Evaluate Transformation K = T^T * K * T
        return np.dot(T.T,np.dot(K,T))
    
    def calcStress(self):
        raise NotImplementedError()
        
    def calcSectionForces(self, dx = 1e9):
        '''
        Eval beam section forces in nodes
        '''
        res = np.zeros((2,7), dtype=float)
            
        prop = self.properties
        mat = self.material
        n1, n2 = self.nodes
        
        L = self.length()
        E = mat['E']
        G = mat['G']
        Ax = prop['Area']
        
        # Eval coordinate transformation matrix
        T = self.calcT()
        
        # Transform displacements from global to local coordinates
        u1 = np.dot(T, n1.results[:3])
        u2 = np.dot(T, n2.results[:3])
        
        # Axial force, Nx
        res[0,0] = 0.
        res[0,1] = (-Ax*E/L)*(u2[0] - u1[0])
        
        res[1,0] = 1.
        res[1,1] = -res[0,1]
                    
        return res
        
class Beam(BaseBeam):
    '''
    2-Node Structural 3D Beam
    '''
    def __init__(self, n1, n2, material = None, properties = None):
        BaseBeam.__init__(self)
        
        self.nodeCount = 2
        self.name = 'Structural 3D Beam'
        
        self.dofSet = DofSet()
        self.dofSet.reset(True)
        
        self.nodes = (n1,n2)
        self.material = material
        self.loads = List()
        
        if properties is None:
            properties = Properties()
        self.properties = properties
    
    def validate(self):
        '''
        Validate beam
        '''
        if self.length() < TINY:
            raise FEError('Beam length is to small')
        
        # check material
        for name in ('E', 'G', 'nu'):
            if name not in self.material:
                raise FEError('Material definition not complete')
            
            if self.material[name] < TINY:
                raise FEError("Material parameter '%s' not correct" % name)
        
        # check profile data
        for name in ('Area', 'Iyy', 'Izz'):
            if name not in self.properties:
                raise FEError('Properties definition not complete')
            
            if self.properties[name] < TINY:
                raise FEError("Properties parameter '%s' not correct" % name)
    
    def calcKe(self):
        '''
        Eval element stiffness matrix in local coordinates.
        
        Includes transverse shear deformation. If no shear deflection
        constant (ShearZ and ShearY) is set, the displacement considers
        bending deformation only. 
        
        Shear deflection constants for other cross-sections can be found in
        structural handbooks.
        '''
        Ke = np.zeros((12,12), dtype=float)
        
        mat = self.material
        prop = self.properties
        
        L = self.length()
        L2 = L*L
        L3 = L2*L
        
        E, G = mat["E"], mat["G"]
        Ax, Iy, Iz = prop["Area"], prop["Iyy"], prop["Izz"]
        
        J = prop.get("Ixx", 0.)
        if J == 0.:
            # Simplification only valid for circular sections
            J = Iy + Iz
            
        Asy = prop.get("ShearY", 0.)
        Asz = prop.get("ShearZ", 0.)
        
        if Asy != 0. and Asz != 0.:
            Ksy = 12.*E*Iz / (G*Asy*L2)
            Ksz = 12.*E*Iy / (G*Asz*L2)
        else:
            Ksy = Ksz = 0.
        
        Ke[0,0]  = Ke[6,6]   = E*Ax / L
        Ke[1,1]  = Ke[7,7]   = 12.*E*Iz / ( L3*(1.+Ksy) )
        Ke[2,2]  = Ke[8,8]   = 12.*E*Iy / ( L3*(1.+Ksz) )
        Ke[3,3]  = Ke[9,9] = G*J / L
        Ke[4,4]  = Ke[10,10] = (4.+Ksz)*E*Iy / ( L*(1.+Ksz) )
        Ke[5,5]  = Ke[11,11] = (4.+Ksy)*E*Iz / ( L*(1.+Ksy) )

        Ke[4,2]  = Ke[2,4]   = -6.*E*Iy / ( L2*(1.+Ksz) )
        Ke[5,1]  = Ke[1,5]   =  6.*E*Iz / ( L2*(1.+Ksy) )
        Ke[6,0]  = Ke[0,6]   = -Ke[0,0]

        Ke[11,7] = Ke[7,11]  =  Ke[7,5] = Ke[5,7] = -Ke[5,1]
        Ke[10,8] = Ke[8,10]  =  Ke[8,4] = Ke[4,8] = -Ke[4,2]
        Ke[9,3] = Ke[3,9]  = -Ke[3,3]
        Ke[10,2] = Ke[2,10]  =  Ke[4,2]
        Ke[11,1] = Ke[1,11]  =  Ke[5,1]

        Ke[7,1]  = Ke[1,7]   = -Ke[1,1]
        Ke[8,2]  = Ke[2,8]   = -Ke[2,2]
        Ke[10,4] = Ke[4,10]  = (2.-Ksz)*E*Iy / ( L*(1.+Ksz) )
        Ke[11,5] = Ke[5,11]  = (2.-Ksy)*E*Iz / ( L*(1.+Ksy) )
        
        return Ke
        
    def calcK(self):
        '''
        Eval element stiffness matrix in global coordinates.
        '''
        K = self.calcKe()
        
        # Eval coordinate transformation matrix
        Tl = self.calcT()
        
        # Build the final transformation matrix - a 12x12 matrix
        T = np.zeros((12,12), dtype=float)
        
        for i in range(4):
            for j in range(3):
                for k in range(3):
                    T[j+3*i,k+3*i] = Tl[j,k]
        
        # Evaluate Transformation K = T^T * K * T
        return np.dot(T.T,np.dot(K,T))
    
    def calcStress(self):
        raise NotImplementedError()
        
    def calcNodalForces(self, res = None):
        '''
        Calculate corresponding global forces at nodes from
        line loads
        '''
        res = self.calcLocalNodalForces(res)
        
        Tl = self.calcT()
        
        # Build the final transformation matrix - a 12x12 matrix
        T = np.zeros((12,12), dtype=float)
        
        for i in range(4):
            for j in range(3):
                for k in range(3):
                    T[j+3*i,k+3*i] = Tl[j,k]
        
        res[:] = np.dot(T.T, res)
        
        return res
    
    def calcLocalNodalForces(self, res = None):
        '''
        Calculate corresponding end forces at nodes from
        line loads in local directions
        '''
        if res is None:
            res = np.zeros((self.dofSet.count() * self.nodeCount,), dtype = float)
        
        L = self.length()
        LL = L * L
        
        load = self.calcLocalLoad()
        load *= L
        
        # Nx
        res[0] += .5*load[0]
        res[6] += .5*load[0]
        # Vy
        res[1] += .5*load[1]
        res[7] += .5*load[1]
        # VZ
        res[2] += .5*load[2]
        res[8] += .5*load[2]
        # My
        res[4] -= load[2] * L / 12.
        res[10] += load[2] * L / 12.
        # Mz
        res[5] -= load[1] * L / 12.
        res[11] += load[1] * L / 12.
        
        return res
    
    def calcSectionForces(self, dx = 1e9):
        # find number of divisions
        try:
            nx = int(self.length()/dx)
            nx = min(max(1, nx), 100)
        except ZeroDivisionError:
            nx = 1
            dx = self.length()
        
        res = np.zeros((nx + 1, 7), dtype=float)
        L = self.length()
        
        # Eval local coordinate transformation matrix
        Tl = self.calcT()
        
        # Build the final transformation matrix - a 12x12 matrix
        T = np.zeros((12,12), dtype=float)
        
        for i in range(4):
            for j in range(3):
                for k in range(3):
                    T[j+3*i,k+3*i] = Tl[j,k]
                    
        # Transform displacements and rotations from global to local coordinates
        n1, n2 = self.nodes
        if n1.cz > n2.cz:
            fac = -1.
        else:
            fac = 1.
            
        d = np.zeros((12,), dtype=float)
        d[:6] = n1.results
        d[6:] = n2.results
        
        u = np.dot(T, d)
        
        # Stiffness matrix in local coordinates
        Ke = self.calcKe()
        
        # Calculate forces in local directions
        forces = np.dot(Ke, u)
        
        # Correct forces from eqivalent forces line loads
        iforces = self.calcLocalNodalForces()
        forces -= iforces
        
        res[0,0] = 0.
        res[0,1:] = forces[:6]
        
        res[nx,0] = 1.
        res[nx,1:] = -forces[6:]
        
        if nx > 1:
            #  accumulate interior span loads
            load = self.calcLocalLoad()
            
            _dx = dx
            x = _dx
            for i in range(nx):
                res[i + 1,0] = x/L                       # Position
                res[i + 1,1] = res[i, 1] + load[0]*_dx   # Axial force, Nx
                res[i + 1,2] = res[i, 2] + load[1]*_dx   # Sheare force Vy
                res[i + 1,3] = res[i, 3] + load[2]*_dx   # Sheare force Vz
                res[i + 1,4] = res[i, 4]                 # Torsion, Txx
                
                # correct last step if needed
                if (i + 1)*_dx > L:
                    _dx = L - i*_dx
                
                x += _dx
                
            # trapezoidal integration of shear force for bending momemnt
            _dx = dx
            for i in range(nx):
                res[i + 1,5] = res[i,5] + .5*(res[i + 1, 3] + res[i, 3])*_dx # Myy
                res[i + 1,6] = res[i,6] + fac*.5*(res[i + 1, 2] + res[i, 2])*_dx # Mzz
                
                # correct last step if needed
                if (i + 1)*_dx > L:
                    _dx = L - i*_dx
            
        return res
        
        
if __name__ == '__main__':
    import math
    from base import FE, Node, BoundCon, CoordSys, LineLoad, localX
    from postprocess import PostProcess
    
    class BeamFE(FE, PostProcess):
        pass
    
    csys = CoordSys().fromEuler132(math.radians(-45.),0.,0.)
    
    fixed = BoundCon(Dx=0.,Dy=0.,Dz=0.,Rx=0.,Ry=0.,Rz=0.)
    pinned = BoundCon(Dx=0.,Dy=0.,Dz=0.)
    roller = BoundCon(Dx=0.)
    free = BoundCon()
    
    n1 = Node(0.,0.,0., boundCon = roller, coordSys = csys)
    #n2 = Node(50.,0.,50., boundCon = BoundCon(Fx=5000.))
    n2 = Node(50.,0.,50., boundCon = free)
    n3 = Node(100.,0.,50., boundCon = free)
    n4 = Node(100.,0.,0., boundCon = fixed) 
    
    prof = Properties(Area=10., Ixx=100., Iyy=200., Izz=300.)
    mat = Material(E = 30E6, G=12E6, nu = 0.4, rho = 7850.)
    
    b1 = Beam(n1, n2, mat, prof)
    b2 = Beam(n2, n3, mat, prof)
    b3 = Beam(n3, n4, mat, prof)
    
    load = LineLoad(-1000., globalZ)
    b2.loads.append(load)
    
    fe = BeamFE((n1,n2,n3,n4),(b1,b2,b3))
    
    fe.validate()
    fe.solve()
    
    fe.printNodalDisp()
    fe.printNodalForces(totals='YES')
    fe.printElementSectionForces(dx=250.)