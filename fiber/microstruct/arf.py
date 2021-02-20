"""
This models one of the microstructured geometries suggested in the paper
[Francesco Poletti. Nested antiresonant nodeless hollow core fiber (2014)].
The fiber is an example of a hollow core Anti Resonant Fiber,
an HC ARF, or ARF as called in the paper.
"""

import netgen.geom2d as geom2d
import ngsolve as ng
from ngsolve import grad, dx
import numpy as np
from pyeigfeast.spectralproj.ngs import NGvecs, SpectralProjNG
from pyeigfeast.spectralproj.ngs import SpectralProjNGGeneral
from fiberamp.fiber.spectralprojpoly import SpectralProjNGPoly
import os
import pickle


class ARF:

    def __init__(self, freecapil=False, **kwargs):
        """
        PARAMETERS:

           freecapil: If True, capillary tubes in the microstructure will
                be modeled as free standing in the hollow region.
                Otherwise, they will be embedded into the glass sheath.

           kwargs: Override default values of updatable length attributes.
                Give length values in units of micrometers.
                A keyword argument 'scaling', if given, will also divide
                the updatable length attributes by 'scaling'.
        """

        self.freecapil = freecapil

        # DEFAULT ATTRIBUTE VALUES

        self.scaling = 15  # to get actual length, multiply by scaling * 1e-6

        # Primary geometrical parameters (geometry shown in tex folder)
        self.Rc = 15.0    # radius of inner part of hollow core
        self.tclad = 5    # thickness of the glass jacket/sheath
        self.touter = 10  # thickness of final outer annular layer
        self.t = 0.42     # thickness of the capillary tubes

        # Attributes for tuning mesh sizes

        self.capillary_maxh = 0.5
        self.air_maxh = 4.0
        self.inner_core_maxh = 1
        self.glass_maxh = 10.0
        self.outer_maxh = 10.0
        self.refined = 0

        # Updatable length attributes. All lengths are in micrometers.

        self.updatablelengths = ['Rc', 'tclad', 't', 'touter',
                                 'capillary_maxh', 'air_maxh',
                                 'inner_core_maxh', 'glass_maxh', 'outer_maxh']

        # Scale updatablelengths and store scaled values in class attributes
        # whose name has an 's' appended. Only the scaled values are used
        # for geometry and mesh construction.
        for key in self.updatablelengths:  # Divide these lengths by scaling
            setattr(self, key + 's', getattr(self, key)/self.scaling)

        # Attributes specific to the embedded capillary case:
        #
        #    e = fraction of the capillary tube thickness that
        #        is embedded into the adjacent silica layer. When
        #        e=0, the outer circle of the capillary tube
        #        osculates the circular boundary of the silica
        #        layer. When e=1, the inner circle of the capillary
        #        tube is tangential to the circular boundary (and
        #        the outer circle is embedded). Value of e must be
        #        strictly greater than 0 and less than or equal to 1.
        self.e = 0.025 / self.t    # nondimensional fraction

        # Attributes used only in the freestanding capillary case:
        #
        #    s = separation of the hollow capillary tubes as a
        #        percentage of the radial distance from the center of
        #        the fiber to the center of capillary tubes (which
        #        would be tangential to the outer glass jacket when s=0).
        self.s = 0.05              # nondimensional fraction

        # Physical parameters

        self.n_air = 1.00027717    # refractive index of air
        self.n_si = 1.4545         # refractive index of glass
        self._wavelength = 1.8e-6  # fiber's operating wavelength

        # UPDATE (any of the above) attributes using given inputs

        for key, value in kwargs.items():
            setattr(self, key, value)

        # create attributes ending in 's' (nondim scaled lengths)
        if 'scaling' in kwargs:    # scale all updatable lengths
            for key in self.updatablelengths:
                setattr(self, key + 's', getattr(self, key)/kwargs['scaling'])

        # attributes in addition to updatablelengths for reconstructing obj
        #    (don't save scaling: avoid re-re-scaling!)
        self.savableattr = ['freecapil', 'n_air', 'n_si', 'wavelength',
                            'e', 's', 'scaling', 'refined']

        # DEPENDENT attributes (dependent on scaled length attributes)

        # distance b/w capillary tubes
        self.ds = 5 * self.ts
        # inner radius of the capillary tubes
        self.Rto = self.Rcs - self.ds
        # outer radius of the capillary tubes
        self.Rti = self.Rto - self.ts

        if self.freecapil:
            # outer radius of glass sheath, where the geometry ends
            self.Rclado = self.Rcs + 2 * self.Rto + self.tclads
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rcs + 2 * self.Rto
        else:
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rcs + self.ts + 2*self.Rti + self.ts*(1-self.e)
            # outer radius of glass sheath
            self.Rclado = self.Rcladi + self.tclads

        # final radius where geometry ends
        self.Rout = self.Rclado + self.touters

        # BOUNDARY & MATERIAL NAMES

        self.material = {
            'Outer': 1,          # outer most annular layer
            'Si': 2,             # cladding & capillaries are glass
            'CapillaryEncl': 3,  # air regions enclosed by capillaries
            'InnerCore': 4,      # inner hollow core (air) region r < Rc
            'FillAir': 5,        # remaining intervening spaces (air)
        }
        mat = self.material
        self.boundary = {
            #              [left domain, right domain]    while going ccw
            'OuterCircle': [mat['Outer'], 0],

            # circle  separating outer most layer from cladding
            'OuterClad':   [mat['Si'], mat['Outer']],

            # inner circular boundary of capillary tubes
            'CapilInner':  [mat['CapillaryEncl'], mat['Si']],

            # artificial inner circular core boundary
            'Inner':       [mat['InnerCore'], mat['FillAir']],

            # outer boundary of capillaries and the inner boundary of
            # sheath/cladding together forms one curve in the case
            # when capillaries are pushed into  the cladding (this curve
            # is absent in the freestanding case):
            'CapilOuterCladInner': [mat['FillAir'], mat['Si']],

            # in the freestanding capillary case, the inner boundary of
            # cladding/sheath is disconnected  from the outer boundaries of
            # capillaries, so we have these two curves (which do not
            # exist in the embedded capillary case).
            'CladInner':  [mat['FillAir'], mat['Si']],
            'CapilOuter': [mat['Si'], mat['FillAir']],
        }

        # CREATE GEOMETRY & MESH

        self.num_capillary_tubes = 6  # only case 6 implemented/tested!

        if self.freecapil:
            self.geo = self.geom_freestand_capillaries()
        else:
            self.geo = self.geom_embedded_capillaries()

        # Set the materials for the domain.
        for material, domain in self.material.items():
            self.geo.SetMaterial(domain, material)

        # Set the maximum mesh sizes in subdomains
        self.geo.SetDomainMaxH(mat['Outer'], self.outer_maxhs)
        self.geo.SetDomainMaxH(mat['Si'], self.glass_maxhs)
        self.geo.SetDomainMaxH(mat['CapillaryEncl'], self.air_maxhs)
        self.geo.SetDomainMaxH(mat['InnerCore'], self.inner_core_maxhs)
        self.geo.SetDomainMaxH(mat['FillAir'], self.air_maxhs)

        # Generate Mesh
        if 'ngmesh' in kwargs:
            print('  Using input mesh.')
            self.mesh = ng.Mesh(kwargs['ngmesh'])
            self.mesh.ngmesh.SetGeometry(self.geo)
        else:
            print('  Generating new mesh.')
            ngmesh = self.geo.GenerateMesh()
            self.mesh = ng.Mesh(ngmesh)

        self.mesh.Curve(3)

        # MATERIAL COEFFICIENTS

        # index of refraction
        index = {'Outer':         self.n_air,
                 'Si':            self.n_si,
                 'CapillaryEncl': self.n_air,
                 'InnerCore':     self.n_air,
                 'FillAir':       self.n_air}
        self.index = ng.CoefficientFunction(
            [index[mat] for mat in self.mesh.GetMaterials()])

        self.setnondimmat()  # coefficient for nondimensionalized eigenproblems

        # OUTPUT LOCATION

        self.outfolder = './outputs'
        if os.path.isdir(self.outfolder) is not True:
            os.mkdir(self.outfolder)

        print('\nInitialized: ', self)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, lam):
        self._wavelength = lam
        self.setnondimmat()

    def setnondimmat(self):
        """ set the material cf """

        a = self.scaling * 1e-6
        k = self.wavenum()
        m = {'Outer':         0,
             'Si':            self.n_si**2 - self.n_air**2,
             'CapillaryEncl': 0,
             'InnerCore':     0,
             'FillAir':       0}
        self.m = ng.CoefficientFunction(
            [(a*k)**2 * m[mat] for mat in self.mesh.GetMaterials()])

    def __str__(self):
        s = 'ARF Physical Parameters:' + \
            '\n  Rc = %g x %g x 1e-6 meters' % (self.Rcs, self.scaling)
        s += '\n  tclad = %g x %g x 1e-6 meters' % (self.tclads, self.scaling)
        s += '\n  touter = %g x %g x 1e-6 meters' % \
            (self.touters, self.scaling)
        s += '\n  t = %g x %g x 1e-6 meters' % (self.ts, self.scaling)
        s += '\n  d = %g x %g x 1e-6 meters' % (self.ds, self.scaling)
        s += '\n  Rti = %g x %g x 1e-6 meters' % (self.Rti, self.scaling)
        s += '\n  Rto = %g x %g x 1e-6 meters' % (self.Rto, self.scaling)
        s += '\n  Wavelength = % g meters' % self.wavelength
        s += '\n  Refractive indices: %g (air), %g (Si)' % \
            (self.n_air, self.n_si)

        s += '\nNondimensional Computational Parameters:'
        s += '\n  Divide all lengths above by %g x 1e-6' % self.scaling
        s += '\n  to get the actual computational lengths used.'
        s += '\n  Mesh sizes: %g (capillary), %g (air), %g (inner core)' \
            % (self.capillary_maxhs, self.air_maxhs, self.inner_core_maxhs)
        s += '\n  Mesh sizes: %g (glass), %g (outer)'  \
            % (self.glass_maxhs,   self.outer_maxhs)
        if self.refined > 0:
            s += '\n  Uniformly refined %g times.' % self.refined
        if self.freecapil:
            s += '\n  With free capillaries, s = %g.' % self.s
        else:
            s += '\n  With embedded capillaries, e/t = %g.' % self.e
        return s

    # GEOMETRY ########################################################

    def geom_freestand_capillaries(self):

        geo = geom2d.SplineGeometry()
        bdr = self.boundary

        # The outermost circle
        geo.AddCircle(c=(0, 0), r=self.Rout,
                      leftdomain=bdr['OuterCircle'][0], rightdomain=0,
                      bc='OuterCircle')

        # The glass sheath
        geo.AddCircle(c=(0, 0), r=self.Rclado,
                      leftdomain=bdr['OuterClad'][0],
                      rightdomain=bdr['OuterClad'][1], bc='OuterClad')
        geo.AddCircle(c=(0, 0), r=self.Rcladi,
                      leftdomain=bdr['CladInner'][0],
                      rightdomain=bdr['CladInner'][1],
                      bc='CladInner')

        # The capillary tubes
        Nxc = 0                   # N tube center xcoord
        Nyc = self.Rcs + self.Rto  # N tube center ycoord
        NEyc = Nyc / 2            # NE tube center ycoord
        NExc = ng.sqrt(Nyc**2-NEyc**2)  # NE tube center xcoord
        NWxc = -NExc             # NW tube center ycoord
        NWyc = NEyc              # NW tube center xcoord
        Sxc = 0                  # S tube center xcoord
        Syc = -Nyc               # S tube center ycoord
        SExc = NExc              # SE tube center xcoord
        SEyc = -NEyc             # SE tube center ycoord
        SWxc = -NExc             # SW tube center ycoord
        SWyc = -NEyc             # SW tube center xcoord

        Nc = (Nxc*(1-self.s), Nyc*(1-self.s))
        NEc = (NExc*(1-self.s), NEyc*(1-self.s))
        NWc = (NWxc*(1-self.s), NWyc*(1-self.s))
        Sc = (Sxc*(1-self.s), Syc*(1-self.s))
        SEc = (SExc*(1-self.s), SEyc*(1-self.s))
        SWc = (SWxc*(1-self.s), SWyc*(1-self.s))

        for c in [Nc, NEc, NWc, Sc, SEc, SWc]:
            geo.AddCircle(c=c, r=self.Rti,
                          leftdomain=bdr['CapilInner'][0],
                          rightdomain=bdr['CapilInner'][1],
                          bc='CapilInner', maxh=self.capillary_maxhs)
            geo.AddCircle(c=c, r=self.Rto,
                          leftdomain=bdr['CapilOuter'][0],
                          rightdomain=bdr['CapilOuter'][1],
                          bc='CapilOuter', maxh=self.capillary_maxhs)

        # Inner core region (not physcial, only used for refinement)
        radius = 0.75 * self.Rcs
        geo.AddCircle(c=(0, 0), r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxhs)

        return geo

    def geom_embedded_capillaries(self):

        # The origin of our coordinate system.
        origin = (0.0, 0.0)
        bdr = self.boundary
        geo = geom2d.SplineGeometry()

        # The outermost circle
        geo.AddCircle(c=(0, 0), r=self.Rout,
                      leftdomain=bdr['OuterCircle'][0], rightdomain=0,
                      bc='OuterCircle')

        # Cladding begins here
        geo.AddCircle(c=(0, 0), r=self.Rclado,
                      leftdomain=bdr['OuterClad'][0],
                      rightdomain=bdr['OuterClad'][1], bc='OuterClad')

        # Inner portion:

        # The angle 'phi' corresponds to the polar angle that gives the
        # intersection of the two circles of radius Rcladi and Rto, resp.
        # The coordinates of the intersection can then be recovered as
        # (Rcladi * cos(phi), Rcladi * sin(phi)) and
        # (-Rcladi * cos(phi), Rcladi * sin(phi)).

        phi = np.arcsin((self.Rcladi**2 +
                         (self.Rcs + self.Rto)**2 - self.Rto**2)
                        / (2 * (self.Rcs + self.Rto) * self.Rcladi))

        # Obtain the angle of the arc between two  capillaries.
        psi = 2 * (phi - np.pi / 3)

        # Get the distance to the middle control pt for the aforementioned arc.
        D = self.Rcladi / np.cos(psi / 2)

        # The center of the top circle.
        c = (0, self.Rcs + self.Rto)

        capillary_points = []

        for k in range(self.num_capillary_tubes):
            # Compute the rotation angle.
            rotation_angle = k * np.pi / 3

            # Compute the middle control point for the outer arc.
            capillary_points += [(D * np.cos(phi - psi / 2 + rotation_angle),
                                  D * np.sin(phi - psi / 2 + rotation_angle))]

            # Obtain the control points for the capillary tube immediately
            # counterclockwise from the above control point.
            capillary_points += \
                self.get_capillary_spline_points(c,  phi, k * np.pi / 3)

        # Add the capillary points to the geometry
        capnums = [geo.AppendPoint(x, y) for x, y in capillary_points]
        NP = len(capillary_points)    # number of capillary point IDs.
        for k in range(1, NP + 1, 2):  # add the splines.
            geo.Append(
                [
                    'spline3',
                    capnums[k % NP],
                    capnums[(k + 1) % NP],
                    capnums[(k + 2) % NP]
                ],
                leftdomain=bdr['CapilOuterCladInner'][0],
                rightdomain=bdr['CapilOuterCladInner'][1],
                bc='CapilOuterCladInner'
            )

        # Add capillary tubes

        # The coordinates of the tube centers
        Nxc = 0         # N tube center xcoord
        Nyc = self.Rcs + self.Rto  # N tube center ycoord
        NEyc = Nyc / 2  # NE tube center ycoord
        NExc = np.sqrt(Nyc**2-NEyc**2)  # NE tube center xcoord
        NWxc = -NExc    # NW tube center ycoord
        NWyc = NEyc     # NW tube center xcoord
        Sxc = 0         # S tube center xcoord
        Syc = -Nyc      # S tube center ycoord
        SExc = NExc     # SE tube center xcoord
        SEyc = -NEyc    # SE tube center ycoord
        SWxc = -NExc    # SW tube center ycoord
        SWyc = -NEyc    # SW tube center xcoord

        Nc = (Nxc,  Nyc)
        NEc = (NExc, NEyc)
        NWc = (NWxc, NWyc)
        Sc = (Sxc,  Syc)
        SEc = (SExc, SEyc)
        SWc = (SWxc, SWyc)

        # The capillary tubes
        for c in [Nc, NEc, NWc, Sc, SEc, SWc]:

            geo.AddCircle(c=c, r=self.Rti,
                          leftdomain=bdr['CapilInner'][0],
                          rightdomain=bdr['CapilInner'][1],
                          bc='CapilInner', maxh=self.capillary_maxhs)

        # Add the circle for the inner core.
        radius = 0.75 * self.Rcs
        geo.AddCircle(c=origin, r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxhs)

        return geo

    def get_capillary_spline_points(self, c, phi, rotation_angle):
        """
        Method that obtains the spline points for the interface between one
        capillary tube and inner hollow core. By default, we generate the
        spline points for the topmost capillary tube, and then rotate
        these points to generate the spline points for another tube based
        upon the inputs.

        INPUTS:

        c  = the center of the northern capillary tube

        phi = corresponds to the polar angle that gives theintersection of
          the two circles of radius Rcladi and Rto, respectively. In this
          case, the latter circle has a center of c given as the first
          argument above.

        rotation_angle = the angle by which we rotate the spline points
          obtained for the north circle to obtain the spline
          points for another capillary tube in the fiber.

        OUTPUTS:

        A list of control point for a spline that describes the interface.
        """

        # Start off with an array of x- and y-coordinates. This will
        # make any transformations to the points easier to work with.
        points = np.zeros((2, 9))

        # Determine the corresponding angle in the unit circle.
        psi = np.arccos((self.Rcladi * np.cos(phi) - c[0]) / self.Rto)

        # The control points for the first spline.
        points[:, 0] = [np.cos(psi), np.sin(psi)]
        points[:, 1] = [1, (1 - np.cos(psi)) / np.sin(psi)]
        points[:, 2] = [1, 0]

        # Control points for the second and third splines.
        points[:, 3] = [1, -1]
        points[:, 4] = [0, -1]
        points[:, 5] = [-1, -1]

        # Control points for the final spline.
        points[:, 6] = [-1, 0]
        points[:, 7] = [-1, (1 - np.cos(psi)) / np.sin(psi)]
        points[:, 8] = [-np.cos(psi), np.sin(psi)]

        # The rotation matrix needed to generate the spline points for an
        # arbitrary capillary tube.
        R = np.mat(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)]
            ]
        )

        # Rotate, scale, and shift the points.
        points *= self.Rto
        points[0, :] += c[0]
        points[1, :] += c[1]
        points = np.dot(R, points)

        # Return the points as a collection of tuples.
        capillary_points = []

        for k in range(0, 9):
            capillary_points += [(points[0, k], points[1, k])]

        return capillary_points

    def refine(self):
        """ Refine mesh by dividing each triangle into four """

        print('  Refining ARF mesh uniformly.')
        self.refined += 1
        self.mesh.ngmesh.Refine()
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(3)

    # EIGENPROBLEM ####################################################

    def wavenum(self):
        """ Return wavenumber, otherwise known as k."""

        return 2 * np.pi / self.wavelength

    def betafrom(self, Z2):
        """ Return physical propagation constants (beta), given
        nondimensional Z² values (input in Z2), per the formula
        β = sqrt(k²n₀² - (Z/a)²). """

        # account for micrometer lengths & any additional scaling in geometry
        a = self.scaling * 1e-6
        k = self.wavenum()
        akn0 = a * k * self.n_air   # a number less whacky in size
        return np.sqrt(akn0**2 - Z2) / a

    def sqrZfrom(self, betas):
        """ Return values of nondimensional Z squared, given physical
        propagation constants betas, ie, return Z² = a² (k²n₀² - β²). """

        a = self.scaling * 1e-6
        k = self.wavenum()
        n0 = self.n_air
        return (a*k*n0)**2 - (a*betas)**2

    def selfadjsystem(self, p):

        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()

        A = ng.BilinearForm(X)
        A += grad(u)*grad(v) * dx - self.m*u*v * dx
        B = ng.BilinearForm(X)
        B += u * v * dx

        with ng.TaskManager():
            A.Assemble()
            B.Assemble()

        return A, B, X

    def autopmlsystem(self, p, alpha=1):

        radial = ng.pml.Radial(rad=self.Rclado,
                               alpha=alpha*1j, origin=(0, 0))
        self.mesh.SetPML(radial, 'Outer')
        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()
        A = ng.BilinearForm(X)
        B = ng.BilinearForm(X)
        A += (grad(u) * grad(v) - self.m * u * v) * dx
        B += u * v * dx
        with ng.TaskManager():
            A.Assemble()
            B.Assemble()
        return A, B, X

    def lineareig(self, p, method='selfadjoint', initdim=5, stop_tol=1e-13,
                  #    LP01, LP11,  LP02
                  ctrs=(5,   12.71, 25.9),
                  radi=(0.1,  0.1,   0.2)):
        """
        Solve a linear eigenproblem to compute mode approximations.

        If method='selfadjoint', then run selfadjoint feast with
        the given centers and radii by solving a Dirichlet Helmholtz
        eigenproblem. Loss factors cannot be computed with this method.

        If method='auto', use NGSolve's mesh PML transformation to formulate
        and solve a linear nonselfadjoint eigenproblem. Eigenvalues will
        generally have imaginary parts, but we usually do not get as
        good accuracy with this method as with the nonlinear method.

        Default paramater values of ctrs and radii are appropriate
        only for an ARF object with default constructor parameters. """

        npts = 8
        Ys = []
        Zs = []
        betas = []

        if method == 'selfadjoint':
            A, B, X = self.selfadjsystem(p)
        elif method == 'auto':
            A, B, X = self.autopmlsystem(p)
        else:
            raise ValueError('Unimplemented method=%s asked of lineareig'
                             % method)

        for rad, ctr in zip(radi, ctrs):
            Y = NGvecs(X, initdim, B.mat)
            Y.setrandom()
            if method == 'selfadjoint':
                P = SpectralProjNG(X, A.mat, B.mat, rad, ctr,
                                   npts, reduce_sym=True)
            else:
                P = SpectralProjNGGeneral(X, A.mat, B.mat, rad, ctr, npts)

            isherm = method == 'selfadjoint'
            Zsqr, Y, history, Yl = P.feast(Y, hermitian=isherm,
                                           stop_tol=stop_tol)
            Ys.append(Y.copy())
            Zs.append(Zsqr)
            betas.append(self.betafrom(Zsqr))

        return Zs, Ys, betas

    def polypmlsystem(self, p, alpha=1):
        """
        Returns AA, B, X, X3:
          AA = list of 4 cubic matrix polynomial coefficients on FE space X
          X3 = three copies of X
          B = Gram matrix of L^2 inner product on X3.
        """
        dx_pml = dx(definedon=self.mesh.Materials('Outer'))
        dx_int = dx(definedon=self.mesh.Materials
                    ('Si|CapillaryEncl|InnerCore|FillAir'))
        R = self.Rclado  # PML starts right after cladding
        s = 1 + 1j * alpha
        x = ng.x
        y = ng.y
        r = ng.sqrt(x*x+y*y) + 0j
        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()
        ux, uy = grad(u)
        vx, vy = grad(v)

        AA = [ng.BilinearForm(X, check_unused=False)]
        AA[0] += (s*r/R) * grad(u) * grad(v) * dx_pml
        AA[0] += s * (r-R)/(R*r*r) * (x*ux+y*uy) * v * dx_pml
        AA[0] += s * (R-2*r)/r**3 * (x*ux+y*uy) * (x*vx+y*vy) * dx_pml
        AA[0] += -s**3 * (r-R)**2/(R*r) * u * v * dx_pml

        AA += [ng.BilinearForm(X)]
        AA[1] += grad(u) * grad(v) * dx_int
        AA[1] += -self.m * u * v * dx_int
        AA[1] += 2 * (r-R)/r**3 * (x*ux+y*uy) * (x*vx+y*vy) * dx_pml
        AA[1] += 1/r**2 * (x*ux+y*uy) * v * dx_pml
        AA[1] += -2*s*s*(r-R)/r * u * v * dx_pml

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[2] += R/s/r**3 * (x*ux+y*uy) * (x*vx+y*vy) * dx_pml
        AA[2] += -R*s/r * u * v * dx_pml

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[3] += -u * v * dx_int

        # A mass matrix for compound space  X x X x X
        X3 = ng.FESpace([X, X, X])
        u0, u1, u2 = X3.TrialFunction()
        v0, v1, v2 = X3.TestFunction()
        B = ng.BilinearForm(X3)
        B += (u0 * v0 + u1 * v1 + u2 * v2) * ng.dx

        with ng.TaskManager():
            B.Assemble()
            for i in range(len(AA)):
                AA[i].Assemble()

        return AA, B, X, X3

    def polyeig(self, p, alpha=1, stop_tol=1e-12, npts=8, initdim=5,
                #    LP01,   LP11,  LP21   LP02
                ctrs=(2.24,  3.57,  4.75,  5.09),
                radi=(0.05,  0.01,  0.01,  0.01)):
        """Solve the Nannen-Wess nonlinear polynomial PML eigenproblem
        to compute modes with losses. A custom polynomial feast uses
        the given centers and radii to search for the modes.

        PARAMETERS:

        p:        polynomial degree of finite elements
        alpha:    PML strength
        stop_tol: quit feast when relative ew diff are smaller than this
        npts:     number of quadrature points in feast
        initdim:  dimension of initial span for feast
        ctrs, radi: repeat feast with a circular contour centered at
                  ctrs[i] of radius radi[i] for each i. Eigenvalues found by
                  feast for each i are returned in output Zs[i], and the
                  corresponding eigenspaces are in span object Ys[i].

        OUTPUTS:  Zs, Ys, betas

        Zs[i] and Ys[i] are as described above, and betas[i] give the
        propagation constants corresponding to nondimensional
        eigenvalues in Zs[i].
        """

        AA, B, X, X3 = self.polypmlsystem(p=p, alpha=alpha)
        Ys = []
        Yls = []
        Zs = []
        betas = []

        for rad, ctr in zip(radi, ctrs):
            Y = NGvecs(X3, initdim, M=B.mat)
            Yl = Y.create()
            Y.setrandom()
            Yl.setrandom()

            def within(z):
                inside1 = abs(ctr - z)**2 < rad**2
                inside2 = z.imag < 0
                return inside1 & inside2

            P = SpectralProjNGPoly(AA, X, radius=rad, center=ctr, npts=npts,
                                   within=within)

            Z, Y, _, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                  stop_tol=stop_tol)
            y = P.first(Y)
            yl = P.last(Yl)
            Ys.append(y.copy())
            Yls.append(yl.copy())
            Zs.append(Z)
            betas.append(self.betafrom(Z**2))

        return Zs, Ys, Yls, betas, P

    # SAVE & LOAD #####################################################

    def save(self, fileprefix):
        """ Save this object so it can be loaded later """

        arffilename = os.path.abspath(self.outfolder+'/'+fileprefix+'_arf.pkl')
        os.makedirs(os.path.dirname(arffilename), exist_ok=True)
        print('Pickling ARF object into ', arffilename)
        with open(arffilename, 'wb') as f:
            pickle.dump(self, f)

    def savemodes(self, fileprefix, Y, p, betas,
                  solverparams, arfpickle=True):
        """ Save a NGVec span object Y containing modes of FE degree p.
        Include any solver paramaters to be saved together with the
        modes in the input dictionary "solverparams".
        """

        if arfpickle:
            self.save(fileprefix)
        y = Y.tonumpy()
        d = {'y': y, 'p': p, 'betas': betas}
        d.update(**solverparams)
        f = os.path.abspath(self.outfolder+'/'+fileprefix+'_mde.npz')
        print('Writing mode file ', f)
        np.savez(f, **d)


# LOAD FROM FILE #####################################################


def loadarf(fileprefix):
    """ Load a saved ARF object from file <fileprefix>_arf.pkl """

    arffile = os.path.abspath(fileprefix+'_arf.pkl')
    with open(arffile, 'rb') as f:
        a = pickle.load(f)
    return a


def loadarfmode(modenpzf, arfpklf):
    """  Load a mode saved in npz file <modenpzf> compatible with the
    ARF object saved in pickle file <arfpklf>. """

    a = loadarf(arfpklf)
    modef = os.path.abspath(modenpzf)
    d = dict(np.load(modef, allow_pickle=True))
    p = int(d.pop('p'))
    betas = d.pop('betas')
    y = d.pop('y')
    for k, v in d.items():
        if v.ndim > 0:
            d[k] = v
        else:  # convert singleton arrays to scalars
            d[k] = v.item()
    print('  Degree %d modes found in file %s' % (p, modef))
    fes = ng.H1(a.mesh, order=p, dirichlet='Outer', complex=True)
    Y = NGvecs(fes, y.shape[1])
    Y.fromnumpy(y)

    return a, Y, betas, p, d
