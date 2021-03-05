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

    def __init__(self, name=None, freecapil=False, **kwargs):
        """
        PARAMETERS:

           name: If None, default to using the ARF microstructure fiber
                with 6 capillary tubes. Otherwise, set geometric fibers
                for the fiber based on the name provided.

           freecapil: If True, capillary tubes in the microstructure will
                be modeled as free standing in the hollow region.
                Otherwise, they will be embedded into the glass sheath.

           kwargs: Override default values of updatable length attributes.
                Give length values in units of micrometers, e.g.,
                    ARF(touter=15)
                yields a PML thickness of 15 micrometers in physical units.
                A keyword argument 'scaling', if given, will also divide
                the updatable length attributes by 'scaling'.
        """

        self.freecapil = freecapil

        # Set the fiber parameters.
        self.set(name=name)

        # Updatable length attributes. All lengths are in micrometers.

        self.updatablelengths = ['Rc', 'Rto', 'Rti', 't', 'd', 'tclad',
                                 'touter']

        # Physical parameters

        self.n_air = 1.00027717    # refractive index of air
        self.n_si = 1.4545         # refractive index of glass

        # UPDATE attributes set in ARF.set() using given inputs

        for key, value in kwargs.items():
            setattr(self, key, value)

        # TODO: Update remaining lengths dependent on the set attributes. If
        # the user updates an arbitrary set of parameters, we need to perform
        # a geometry check to make sure that the parameters the user set are
        # not erroneous.

        # Set the scaling as requested by the user.
        if 'scaling' in kwargs:
            self.scaling = kwargs['scaling']

        # Scale updatablelengths and store scaled values in class attributes
        # whose name has an 's' appended. Only the scaled values are used
        # for geometry and mesh construction.
        for key in self.updatablelengths:
            setattr(self, key + 's', getattr(self, key)/self.scaling)

        # attributes in addition to updatablelengths for reconstructing obj
        #    (don't save scaling: avoid re-re-scaling!)
        self.savableattr = ['freecapil', 'n_air', 'n_si', 'wavelength',
                            'e', 's', 'scaling', 'refined']

        if self.freecapil:
            # outer radius of glass sheath, where the geometry ends
            self.Rclado = self.Rcs + 2 * self.Rtos + self.tclads
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rcs + 2 * self.Rtos
        else:
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rcs + self.ts + 2*self.Rtis + self.ts*(1-self.e)
            # outer radius of glass sheath
            self.Rclado = self.Rcladi + self.tclads

        # final radius where geometry ends
        self.Rout = self.Rclado + self.touters

        # BOUNDARY & MATERIAL NAMES

        self.material = {
            'Outer': 1,          # outer most annular layer (PML)
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

        # Before creating the mesh, check to make sure that the geometric
        # parameters lead to a mesh with non-tangent and non-overlapping
        # subdomains (overlapping and tangent subdomains can cause netgen
        # to crash).
        self._check_geometric_parameters()

        # CREATE GEOMETRY & MESH
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

    def set(self, name=None):
        """
        Method that sets the geometric parameters of the ARF fiber based
        on the name supplied by the user. Specifying name='poletti'
        yields a six-capillary fiber, while name='kolyadin' specifies
        an 8-capillary fiber.

        Attributes specific to the embedded capillary case:

           e = fraction of the capillary tube thickness that
               is embedded into the adjacent silica layer. When
               e=0, the outer circle of the capillary tube
               osculates the circular boundary of the silica
               layer. When e=1, the inner circle of the capillary
               tube is tangential to the circular boundary (and
               the outer circle is embedded). Value of e must be
               strictly greater than 0 and less than or equal to 1.

        Attributes used only in the freestanding capillary case:

           s = separation of the hollow capillary tubes as a
               percentage of the radial distance from the center of
               the fiber to the center of capillary tubes (which
               would be tangential to the outer glass jacket when s=0).
        """

        # By default, we'll use the 6-capillary fiber.
        self.name = 'poletti' if name is None else name

        if self.name == 'poletti':
            # This case gives the default attributes of the fiber.

            self.Rc = 15                  # core radius
            self.Rto = 12.9               # capillary outer radius
            self.Rti = 12.48              # capillary inner radius
            self.t = self.Rto - self.Rti  # capillary thickness
            self.tclad = 5                # glass jacket (cladding) thickness
            self.touter = 30              # outer jacket (PML) thickness
            self.scaling = self.Rc        # scaling for the PDE
            self.num_capillary_tubes = 6  # number of capillaries
            self.s = 0.05
            self.e = 0.025 / self.t
            self._wavelength = 1.8e-6

            # Compute the capillary tube separation distance.
            half_sector = np.pi / self.num_capillary_tubes
            D = 2 * (self.Rc + self.Rto) * np.sin(half_sector)
            self.d = D - 2 * self.Rto

            # Set the (non-dimensional) mesh sizes.
            self.capillary_maxhs = 0.05
            self.air_maxhs = 0.25
            self.inner_core_maxhs = 0.25
            self.glass_maxhs = 0.33
            self.outer_maxhs = 2.0

            self.refined = 0
        elif self.name == 'kolyadin':
            self.Rc = 59.5                # core radius
            self.Rto = 31.5               # capillary outer radius
            self.Rti = 25.5               # capillary inner radius
            self.t = self.Rto - self.Rti  # capillary thickness
            self.tclad = 1.2 * self.Rti   # glass jacket (cladding) thickness
            self.touter = 30              # outer jacket (PML) thickness
            self.scaling = self.Rc        # scaling for the PDE
            self.num_capillary_tubes = 8  # number of capillaries
            self.s = 0.05
            self.e = 2.0 / self.t
            self._wavelength = 5.75e-6

            # Compute the capillary tube separation distance.
            half_sector = np.pi / self.num_capillary_tubes
            D = 2 * (self.Rc + self.Rto) * np.sin(half_sector)
            self.d = D - 2 * self.Rto

            # Set the (non-dimensional) mesh sizes.
            self.capillary_maxhs = 0.04
            self.air_maxhs = 0.25
            self.inner_core_maxhs = 0.25
            self.glass_maxhs = 0.5
            self.outer_maxhs = 0.5

            self.refined = 0
        else:
            err_str = 'Fiber \'{:s}\' not implemented.'.format(self.name)
            raise NotImplementedError(err_str)

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
        s += '\n  Wavelength = %g meters' % self.wavelength
        s += '\n  Refractive indices: %g (air), %g (Si)' % \
            (self.n_air, self.n_si)

        s += '\nNondimensional Computational Parameters:'
        s += '\n  Divide all lengths above by %g x 1e-6' % self.scaling
        s += '\n  to get the actual computational lengths used.'
        s += '\n  Cladding starts at Rcladi = %g' % self.Rcladi
        s += '\n  PML starts at Rclado = %g and ends at Rout = %g' \
            % (self.Rclado, self.Rout)
        s += '\n  Mesh sizes: %g (capillary), %g (air), %g (inner core)' \
            % (self.capillary_maxhs, self.air_maxhs, self.inner_core_maxhs)
        s += '\n  Mesh sizes: %g (glass), %g (outer)'  \
            % (self.glass_maxhs,   self.outer_maxhs)
        s += '\n  Elements/wavelength:'
        s += '%g (capillary), %g (air), %g (inner core)' \
            % (self.wavelength*1e6/self.scaling/self.capillary_maxhs,
               self.wavelength*1e6/self.scaling/self.air_maxhs,
               self.wavelength*1e6/self.scaling/self.inner_core_maxhs)
        s += '\n  Elements/wavelength: %g (glass), %g (outer)'  \
            % (self.wavelength*1e6/self.scaling/self.glass_maxhs,
               self.wavelength*1e6/self.scaling/self.outer_maxhs)
        if self.refined > 0:
            s += '\n  Uniformly refined %g times.' % self.refined
        if self.freecapil:
            s += '\n  With free capillaries, s = %g.' % self.s
        else:
            s += '\n  With embedded capillaries, e/t = %g.' % self.e
        return s

    # GEOMETRY ########################################################

    def _check_geometric_parameters(self):
        """
        Method that checks to make sure there are no overlapping
        capillary tubes, or that other parameters provided for the
        geometry are not erroneous.
        """

        if self.freecapil:
            # First, given the current geometric parameters, compute
            # an upper bound on the capillary contraction parameters.
            half_sector = np.pi / self.num_capillary_tubes
            frac = self.Rtos / ((self.Rcs + self.Rtos) * np.sin(half_sector))
            sub = 1 - frac

            # Next, given the current geometric parameters, compute
            # an upper bound on the number of capillary tubes and
            # check the current specified number of capillary tubes
            # against this.
            asin_frac = self.Rtos / ((1 - self.s) * (self.Rcs + self.Rtos))
            nub = int(np.floor(np.pi / np.arcsin(asin_frac)))

            # A placeholder for the number of user-specified tubes.
            n = self.num_capillary_tubes

            if n > nub:
                # Set a new upper bound on s that uses the maximum lower
                # bound on the number of capillary tubes.
                frac = self.Rtos / ((self.Rcs + self.Rtos) *
                                    np.sin(np.pi / nub))
                new_sub = sub if sub > 0 else 1 - frac

                err_str = 'Specifying {0:d} capillary tube(s) '.format(n) \
                    + 'results in tangent or overlapping capillary' \
                    + 'subdomains. Consider setting ' \
                    + '\'num_capillary_tubes\' less than or equal ' \
                    + 'to {:d} and then'.format(nub) \
                    + 'setting s < {:.3f},'.format(new_sub) \
                    + 'or adjusting other geometric parameters.'
        else:
            # For the embedded case, we first check to make sure the embed
            # fraction e gives us a resulting geometry that doesn't have
            # tangent subdomain boundaries (e = 0) or capillary tubes that
            # get pushed into the outer glass jacket.
            if self.e <= 0 or self.e > 1:
                err_str = 'Current value of e = {:.3f}'.format(self.e) \
                        + 'results in capillary tubes in invalid ' \
                        + 'positions. The embedding fraction \'e\'' \
                        + 'must be a real number satisfying ' \
                        + '0 < e <= 1.'
                raise ValueError(err_str)

            # Next, given the current geometric parameters, compute
            # an upper limit on the number of capillary tubes and
            # check the current specified number of capillary tubes
            # against this.
            asin_frac = self.Rtos / (self.Rcs + self.Rtos)
            nub = int(np.floor(np.pi / np.arcsin(asin_frac)))

            # A placeholder for the number of user-specified tubes.
            n = self.num_capillary_tubes

            if n > nub:
                err_str = 'Specifying {0:d} capillary tube(s) '.format(n) \
                        + 'results in tangent or overlapping capillary' \
                        + 'subdomains. Consider making ' \
                        + '\'num_capillary_tubes\' less than or equal to ' \
                        + '{0:d}'.format(nub) \
                        + 'or adjusting other geometric parameters.'
                raise ValueError(err_str)

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

        # --------------------------------------------------------------------
        # Add capillary tubes.
        # --------------------------------------------------------------------

        # Spacing for the angles we need to add the inner circles for the
        # capillaries.
        theta = np.pi / 2.0 + np.linspace(0, 2*np.pi,
                                          num=self.num_capillary_tubes,
                                          endpoint=False)

        # The radial distance to the capillary tube centers.
        dist = (1 - self.s) * (self.Rcs + self.Rtos)

        for t in theta:
            c = (dist*np.cos(t), dist*np.sin(t))

            geo.AddCircle(c=c, r=self.Rtis,
                          leftdomain=bdr['CapilInner'][0],
                          rightdomain=bdr['CapilInner'][1],
                          bc='CapilInner', maxh=self.capillary_maxhs)

            geo.AddCircle(c=c, r=self.Rtos,
                          leftdomain=bdr['CapilOuter'][0],
                          rightdomain=bdr['CapilOuter'][1],
                          bc='CapilOuter', maxh=self.capillary_maxhs)

        # Add the circle for the inner core. Since we are scaling back the
        # (original) distance to the capillary tube centers by (1 - s), we
        # necessarily need to do the same for the inner core region.
        radius = 0.75 * self.Rcs * (1 - self.s)
        geo.AddCircle(c=(0, 0), r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxhs)

        return geo

    def geom_embedded_capillaries(self):
        # Grab the boundary dictionary and create the spline geometry.
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

        numerator = self.Rcladi**2 + (self.Rcs + self.Rtos)**2 - self.Rtos**2
        denominator = 2 * (self.Rcs + self.Rtos) * self.Rcladi
        acos_frac = numerator / denominator
        phi = np.arccos(acos_frac)

        # The angle of a given sector that bisects two adjacent capillary
        # tubes. Visually, this looks like a wedge in the computational domain
        # that contains a half capillary tube on each side of the widest part
        # of the wedge.
        sector = 2 * np.pi / self.num_capillary_tubes

        # Obtain the angle of the arc between two capillaries. This subtends
        # the arc between the two points where two adjacent capillary tubes
        # embed into the outer glass jacket.
        psi = sector - 2 * phi

        # Get the distance to the middle control point for the aforementioned
        # arc.
        D = self.Rcladi / np.cos(psi / 2)

        # The center of the top capillary tube.
        c = (0, self.Rcs + self.Rtos)

        capillary_points = []

        for k in range(self.num_capillary_tubes):
            # Compute the rotation angle needed for rotating the north
            # capillary spline points to the other capillary locations in the
            # domain.
            rotation_angle = k * sector

            # Compute the middle control point for the outer arc subtended by
            # the angle psi + rotation_angle.
            ctrl_pt_angle = np.pi / 2 - phi - psi / 2 + rotation_angle
            capillary_points += [(D * np.cos(ctrl_pt_angle),
                                  D * np.sin(ctrl_pt_angle))]

            # Obtain the control points for the capillary tube immediately
            # counterclockwise from the above control point.
            capillary_points += \
                self.get_capillary_spline_points(c, phi, rotation_angle)

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

        # --------------------------------------------------------------------
        # Add capillary tubes.
        # --------------------------------------------------------------------

        # Spacing for the angles we need to add the inner circles for the
        # capillaries.
        theta = np.pi / 2.0 + np.linspace(0, 2*np.pi,
                                          num=self.num_capillary_tubes,
                                          endpoint=False)

        # The radial distance to the capillary tube centers.
        dist = self.Rcs + self.Rtos

        for t in theta:
            c = (dist*np.cos(t), dist*np.sin(t))

            geo.AddCircle(c=c, r=self.Rtis,
                          leftdomain=bdr['CapilInner'][0],
                          rightdomain=bdr['CapilInner'][1],
                          bc='CapilInner', maxh=self.capillary_maxhs)

        # Add the circle for the inner core.
        radius = 0.75 * self.Rcs
        geo.AddCircle(c=(0, 0), r=radius,
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

        # Compute the angle inside of the capillary tube to determine some
        # of the subsequent spline points for the upper half of the outer
        # capillary tube.
        acos_frac = (self.Rcladi * np.sin(phi) - c[0]) / self.Rtos
        psi = np.arccos(acos_frac)

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
        points *= self.Rtos
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

        print('  Refining ARF mesh uniformly: each element split into four')
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

        X = ng.H1(self.mesh, order=p, complex=True)

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
        X = ng.H1(self.mesh, order=p, complex=True)
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
        X = ng.H1(self.mesh, order=p, complex=True)
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

        # A mass matrix for X
        u, v = X.TnT()
        B = ng.BilinearForm(X)
        B += u * v * dx

        with ng.TaskManager():
            B.Assemble()
            for i in range(len(AA)):
                AA[i].Assemble()

        X3 = ng.FESpace([X, X, X])
        B3 = ng.BlockMatrix([[B.mat, None, None],
                             [None, B.mat, None],
                             [None, None, B.mat]])

        return AA, B.mat, X, B3, X3

    def polyeig(self, p, alpha=10, stop_tol=1e-12, npts=8, initdim=5,
                #    LP01,   LP11,  LP21   LP02
                ctrs=(2.24,  3.57,  4.75,  5.09),
                radi=(0.05,  0.01,  0.01,  0.01), **kwargs):
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
        kwargs: further keyword arguments passed to spectral projector.

        OUTPUTS:  Zs, Ys, betas

        Zs[i] and Ys[i] are as described above, and betas[i] give the
        propagation constants corresponding to nondimensional
        eigenvalues in Zs[i].
        """

        AA, B, X, B3, X3 = self.polypmlsystem(p=p, alpha=alpha)
        print('Set PML with alpha=', alpha, 'and thickness=%.3f'
              % self.touters)
        Ys = []
        longYs = []
        Yls = []
        longYls = []
        Zs = []
        betas = []

        for rad, ctr in zip(radi, ctrs):
            Y = NGvecs(X3, initdim, M=B3)
            Yl = Y.create()
            Y.setrandom(seed=1)
            Yl.setrandom(seed=1)

            def within(z):
                # look below the real axis only
                inside1 = abs(ctr - z)**2 < rad**2
                inside2 = z.imag < 0
                return inside1 & inside2

            P = SpectralProjNGPoly(AA, X, radius=rad, center=ctr, npts=npts,
                                   within=within, **kwargs)

            Z, Y, _, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                  stop_tol=stop_tol)
            y = P.first(Y)
            yl = P.last(Yl)
            y.centernormalize(self.mesh(0, 0))
            yl.centernormalize(self.mesh(0, 0))
            print('Computed Z =', Z)

            # a posteriori checks
            decayrate = alpha * (self.Rout - self.Rclado) + \
                self.Rclado * Z.imag
            bdryval = np.exp(-decayrate) / np.sqrt(np.abs(Z)*np.pi/2)
            bdrnrm0 = bdryval*2*np.pi*self.Rout
            print('PML guessed boundary norm ~ %.1e' % max(bdrnrm0))
            if np.max(bdrnrm0) > 1e-6:
                print('*** Not enough PML decay for this Z!')

            def outint(u):
                out = self.mesh.Boundaries('OuterCircle')
                s = ng.Integrate(u*ng.Conj(u), out, ng.BND).real
                return ng.sqrt(s)

            bdrnrm = y.applyfnl(outint)
            print('Actual boundary norm = %.1e' % max(bdrnrm))
            if np.max(bdrnrm) > 1e-6:
                print('*** Mode has not decayed in PML enough!')

            Ys.append(y.copy())
            Yls.append(yl.copy())
            longYs.append(Y.copy())
            longYls.append(Yl.copy())
            Zs.append(Z)
            betas.append(self.betafrom(Z**2))

        return Zs, Ys, Yls, betas, P, longYs, longYls

    # SAVE & LOAD #####################################################

    def save(self, fileprefix):
        """ Save this object so it can be loaded later """

        arffilename = os.path.abspath(self.outfolder+'/'+fileprefix+'_arf.pkl')
        os.makedirs(os.path.dirname(arffilename), exist_ok=True)
        print('Pickling ARF object into ', arffilename)
        with open(arffilename, 'wb') as f:
            pickle.dump(self, f)

    def savemodes(self, fileprefix, Y, p, betas, Zs,
                  solverparams, longY=None, longYl=None, arfpickle=False):
        """
        Save a NGVec span object Y containing modes of FE degree p.
        Include any solver paramaters to be saved together with the
        modes in the input dictionary "solverparams". If "arfpickle"
        is True, then the arf object is also save under the same "fileprefix".
        """

        if arfpickle:
            self.save(fileprefix)
        y = Y.tonumpy()
        if longY is not None:
            longY = longY.tonumpy()
        if longYl is not None:
            longYl = longYl.tonumpy()
        d = {'y': y, 'p': p, 'betas': betas, 'Zs': Zs,
             'longy': longY, 'longyl': longYl}
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


def loadarfmode(modenpzf, arffprefix):
    """  Load a mode saved in npz file <modenpzf> compatible with the
    ARF object saved in pickle file <arffprefix>_arf.pkl. """

    a = loadarf(arffprefix)
    modef = os.path.abspath(modenpzf)
    d = dict(np.load(modef, allow_pickle=True))
    p = int(d.pop('p'))
    betas = d.pop('betas')
    Zs = d.pop('Zs')
    y = d.pop('y')
    for k, v in d.items():
        if v.ndim > 0:
            d[k] = v
        else:  # convert singleton arrays to scalars
            d[k] = v.item()
    print('  Degree %d modes found in file %s' % (p, modenpzf))
    X = ng.H1(a.mesh, order=p, complex=True)
    X3 = ng.FESpace([X, X, X])
    Y = NGvecs(X, y.shape[1])
    Y.fromnumpy(y)
    longY = None
    longYl = None
    longy = d['longy']
    longyl = d['longyl']

    if longy is not None:
        longY = NGvecs(X3, longy.shape[1])
        longY.fromnumpy(longy)
    if longyl is not None:
        longYl = NGvecs(X3, longyl.shape[1])
        longYl.fromnumpy(longyl)

    return a, Y, betas, Zs, p, d, longY, longYl
