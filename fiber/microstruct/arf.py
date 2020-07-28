"""
This models one of the microstructured geometries suggested in the paper
[Francesco Poletti. Nested antiresonant nodeless hollow core fiber (2014)].
The fiber is an example of a hollow core Anti Resonant Fiber,
an HC ARF, or ARF as called in the paper.
"""

import netgen.geom2d as geom2d
import ngsolve as ng
import numpy as np


class ARF:

    def __init__(self, freecapil=False, **kwargs):
        """
        PARAMETERS:

           freecapil: If True, capillary tubes in the microstructure will
                      be modeled as free standing in the hollow region.
                      Otherwise, they will be embedded into the glass sheath.

           kwargs: Overwrite default attribute values.
        """

        # DEFAULT ATTRIBUTE VALUES

        self.freecapil = freecapil
        self.scaling = 1

        # Primary geometrical parameters (geometry shown in tex folder)

        self.Rc = 15.0   # radius of inner part of hollow core
        self.tclad = 20  # thickness of the glass jacket/sheath
        self.t = 0.55    # thickness of the capillary tubes

        # Attributes for tuning mesh sizes

        self.capillary_maxh = 0.8
        self.air_maxh = 8.0
        self.inner_core_maxh = 2
        self.glass_maxh = 10.0

        # Updatable length dimensions, usually given in micrometers
        self.updatablelengths = ['Rc', 'tclad', 't', 'capillary_maxh',
                                 'air_maxh', 'inner_core_maxh', 'glass_maxh']

        # Attributes specific to the embedded capillary case:
        #
        #    e = fraction of the capillary tube thickness that
        #        is embedded into the adjacent silica layer. When
        #        embed=0, the outer circle of the capillary tube
        #        osculates the circular boundary of the silica
        #        layer. When embed=1, the inner circle of the capillary
        #        tube is tangential to the circular boundary (and
        #        the outer circle is embedded). Value of embed must be
        #        strictly greater than 0 and less than or equal to 1.
        self.e = 0.017 / self.t    # nondimensional fraction

        # Attributes used only in the freestanding capillary case:
        #
        #    s = separation of the hollow capillary tubes as a
        #        percentage of the radial distance from the center of
        #        the fiber to the center of capillary tubes (which
        #        would be tangential to the outer glass jacket when s=0).
        self.s = 0.05              # nondimensional fraction

        # UPDATE (any of the above) attributes using given inputs

        for key, value in kwargs.items():
            setattr(self, key, value)
        if 'scaling' in kwargs:    # scale all updatable lengths
            for key in self.updatablelengths:
                setattr(self, key, getattr(self, key)/kwargs['scaling'])

        # DEPENDENT attributes

        # distance b/w capillary tubes
        self.d = 5 * self.t
        # inner radius of the capillary tubes
        self.Rto = self.Rc - self.d
        # outer radius of the capillary tubes
        self.Rti = self.Rto - self.t

        if self.freecapil:
            # outer radius of glass sheath, where the geometry ends
            self.Rclado = self.Rc + 2 * self.Rto + self.tclad
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rc + 2 * self.Rto
        else:
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rc + self.t + 2*self.Rti + self.t*(1-self.e)
            # outer radius of glass sheath, where the geometry ends
            self.Rclado = self.Rcladi + self.tclad

        # BOUNDARY & MATERIAL NAMES

        self.material = {
            'Si': 1,             # outer cladding & capillaries are glass
            'CapillaryEncl': 2,  # air regions enclosed by capillaries
            'InnerCore': 3,      # inner hollow core (air) region r < Rc
            'FillAir': 4,        # remaining intervening spaces (air)
        }
        mat = self.material
        self.boundary = {
            #             [left domain, right domain]    while going ccw
            'Outer':      [mat['Si'], 0],

            # inner circular boundary of capillary tubes
            'CapilInner': [mat['CapillaryEncl'], mat['Si']],

            # artificial inner circular core boundary
            'Inner':      [mat['InnerCore'], mat['FillAir']],

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
        self.geo.SetDomainMaxH(mat['Si'], self.glass_maxh)
        self.geo.SetDomainMaxH(mat['CapillaryEncl'], self.air_maxh)
        self.geo.SetDomainMaxH(mat['InnerCore'], self.inner_core_maxh)
        self.geo.SetDomainMaxH(mat['FillAir'], self.air_maxh)

        # Generate Mesh
        ngmesh = self.geo.GenerateMesh()
        self.mesh = ng.Mesh(ngmesh)
        self.mesh.Curve(3)

    def geom_freestand_capillaries(self):

        geo = geom2d.SplineGeometry()
        bdr = self.boundary

        # The glass sheath
        geo.AddCircle(c=(0, 0), r=self.Rclado,
                      leftdomain=bdr['Outer'][0],
                      rightdomain=0, bc='Outer')
        geo.AddCircle(c=(0, 0), r=self.Rcladi,
                      leftdomain=bdr['CladInner'][0],
                      rightdomain=bdr['CladInner'][1],
                      bc='CladInner')

        # The capillary tubes
        Nxc = 0                   # N tube center xcoord
        Nyc = self.Rc + self.Rto  # N tube center ycoord
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
                          bc='CapilInner', maxh=self.capillary_maxh)
            geo.AddCircle(c=c, r=self.Rto,
                          leftdomain=bdr['CapilOuter'][0],
                          rightdomain=bdr['CapilOuter'][1],
                          bc='CapilOuter', maxh=self.capillary_maxh)

        # Inner core region (not physcial, only used for refinement)
        radius = 0.75 * self.Rc
        geo.AddCircle(c=(0, 0), r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxh)

        return geo

    def geom_embedded_capillaries(self):

        # The origin of our coordinate system.
        origin = (0.0, 0.0)
        bdr = self.boundary
        geo = geom2d.SplineGeometry()

        # Create the outermost circle.
        geo.AddCircle(c=origin, r=self.Rclado,
                      leftdomain=bdr['Outer'][0],
                      rightdomain=bdr['Outer'][1], bc='Outer')

        # Inner portion:

        # The angle 'phi' corresponds to the polar angle that gives the
        # intersection of the two circles of radius Rcladi and Rto, resp.
        # The coordinates of the intersection can then be recovered as
        # (Rcladi * cos(phi), Rcladi * sin(phi)) and
        # (-Rcladi * cos(phi), Rcladi * sin(phi)).

        phi = np.arcsin((self.Rcladi**2 +
                         (self.Rc + self.Rto)**2 - self.Rto**2)
                        / (2 * (self.Rc + self.Rto) * self.Rcladi))

        # Obtain the angle of the corresponding arc that sits between two
        # capillaries.
        psi = 2 * (phi - np.pi / 3)

        # Get the distance to the middle control pt for the aforementioned arc.
        D = self.Rcladi / np.cos(psi / 2)

        # The center of the top circle.
        c = (0, self.Rc + self.Rto)

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
        Nyc = self.Rc + self.Rto  # N tube center ycoord
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
                          bc='CapilInner', maxh=self.capillary_maxh)

        # Add the circle for the inner core.
        radius = 0.75 * self.Rc
        geo.AddCircle(c=origin, r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxh)

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


if __name__ == '__main__':

    a = ARF(freecapil=True, scaling=15)

    ng.Draw(a.mesh)
