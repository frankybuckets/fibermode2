import netgen.geom2d as geom2d
import ngsolve as ng
import numpy as np
from fiberamp.fiber.modesolver import ModeSolver


class PBG(ModeSolver):

    def __init__(self, fiber_param_dict):

        for key, value in fiber_param_dict.items():
            setattr(self, key, value)

        if self.scale == 0:
            raise ValueError("Scaling set to zero will yield infinite domain.")

        self.k = 2 * np.pi / self.wavelength

        # Set non-dimensional radii for geometry

        self.R0 = self.S / self.scale
        self.R = (self.S + self.t_air) / self.scale
        self.Rout = (self.S + self.t_air + self.t_outer) / self.scale

        # Create geometry

        self.geo = self.geometry(self.sep, self.r, self.R0, self.R, self.Rout,
                                 self.layers, self.skip,  self.p, self.scale,
                                 self.pattern)

        # Set the materials for the domain.

        self.mat = {
            'Outer': 4,
            'air': 3,
            'tube': 2,
            'clad': 1,
        }

        for material, domain in self.mat.items():
            self.geo.SetMaterial(domain, material)

        # Set the maximum mesh sizes in subdomains

        self.geo.SetDomainMaxH(self.mat['Outer'], self.pml_maxh)
        self.geo.SetDomainMaxH(self.mat['air'], self.air_maxh)
        self.geo.SetDomainMaxH(self.mat['tube'], self.tube_maxh)
        self.geo.SetDomainMaxH(self.mat['clad'], self.clad_maxh)

        self.mesh = ng.Mesh(self.geo.GenerateMesh())
        self.mesh.Curve(3)

        # Initialize parent class ModeSolver

        super().__init__(self.mesh, self.scale, self.n0)

        # Set V function

        m = {'Outer':     0,
             'clad':  self.n0**2 - self.n_clad**2,
             'tube':    self.n0**2 - self.n_tube**2,
             'air':       0}

        self.V = ng.CoefficientFunction(
            [(self.scale*self.k)**2 * m[mat] for mat in
             self.mesh.GetMaterials()])

    def geometry(self, sep, r, R0, R_air, Rout,
                 layers=6, skip=1, p=6, scale=10**6, pattern=None):
        """Construct and return non-dimensionalized geometry."""

        geo = geom2d.SplineGeometry()

        # Non-Dimensionalize needed physical parameters
        sep /= scale
        r /= scale

        for i in range(layers):

            index_layer = skip + i
            R = index_layer * sep

            if pattern is not None:
                self.add_layer(
                    geo, r, R, p=p, innerpoints=index_layer - 1,
                    mask=pattern[i])
            else:
                self.add_layer(geo, r, R, p=p, innerpoints=index_layer - 1)

        # create boundary of fiber
        geo.AddCircle(c=(0, 0), r=R0, leftdomain=1, rightdomain=3)
        geo.AddCircle(c=(0, 0), r=R_air, leftdomain=3,
                      rightdomain=4)    # air/pml boundary
        geo.AddCircle(c=(0, 0), r=Rout, leftdomain=4,
                      bc="OuterCircle")  # outermost circle

        return geo

    def add_layer(self, geo, r, R, p=6, innerpoints=0, mask=None):
        """Add a single layer of small circles of radius 'r' at vertices of
        'p'-sided polygon inscribed in circle of radius 'R' to geometry 'geo'.
        Add circles along edges of polygon by specifying 'innerpoints' and use
        'mask' to create pattern."""

        if p == 0 or p == 1:
            raise ValueError("Please specify a p of 2 or greater.")

        if R == 0:            # zero big radius implies single point at origin
            print('adding circle at origin')
            geo.AddCircle(c=(0, 0), r=r, leftdomain=2, rightdomain=1)

        else:
            xs = []
            ys = []
            for i in range(p):    # get vertex points of polygon
                x0, y0 = R * np.cos(i * 2 * np.pi / p), R * \
                    np.sin(i * 2 * np.pi / p)
                x1, y1 = R * np.cos((i + 1) * 2 * np.pi / p), R * \
                    np.sin((i + 1) * 2 * np.pi / p)

                for s in range(innerpoints + 1):  # interpolate innerpoints
                    t = s * 1/(innerpoints + 1)
                    xs.append((1-t) * x0 + t * x1)
                    ys.append((1-t) * y0 + t * y1)

            if mask is not None:
                # mask out unwanted points
                xs = list(np.array(xs)[np.where(mask)])
                ys = list(np.array(ys)[np.where(mask)])

            for x, y in zip(xs, ys):
                geo.AddCircle(c=(x, y), r=r, leftdomain=2, rightdomain=1)
