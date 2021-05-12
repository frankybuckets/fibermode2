import netgen.geom2d as geom2d
import ngsolve as ng
import numpy as np
import os
import pickle
from fiberamp.fiber.modesolver import ModeSolver
from pyeigfeast.spectralproj.ngs import NGvecs


class PBG(ModeSolver):

    def __init__(self, fiber_param_dict, outfolder='/home/pv/local/fiberamp/fiber/microstruct/pbg/outputs'):

        for key, value in fiber_param_dict.items():
            setattr(self, key, value)

        self.n0 = self.n_air
        self.outfolder = outfolder

        # Create geometry
        self.geo = self.geometry(self.sep, self.r_tube, self.R_fiber, self.R,
                                 self.Rout, self.layers, self.skip,  self.p,
                                 self.scale, self.pattern)

        # Set the materials for the domain.

        self.mat = {5: 'Outer', 4: 'air', 3: 'tube', 2: 'clad', 1: 'core'}

        for domain, material in self.mat.items():
            self.geo.SetMaterial(domain, material)

        # Set the maximum mesh sizes in subdomains

        self.geo.SetDomainMaxH(1, self.core_maxh)
        self.geo.SetDomainMaxH(2, self.clad_maxh)
        self.geo.SetDomainMaxH(3, self.tube_maxh)
        self.geo.SetDomainMaxH(4, self.air_maxh)
        self.geo.SetDomainMaxH(5, self.pml_maxh)

        self.mesh = ng.Mesh(self.geo.GenerateMesh())
        self.mesh.Curve(3)

        # Initialize parent class ModeSolver

        super().__init__(self.mesh, self.scale, self.n0)

        # Set V function

        self.V = ng.CoefficientFunction(
            [(self.scale * self.k) ** 2 *
             (self.n_base ** 2 - self.n_dict[mat] ** 2)
             for mat in self.mesh.GetMaterials()])

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, lam):
        self._wavelength = lam
        self.k = 2 * np.pi / self._wavelength

    def geometry(self, sep, r, R0, R_air, Rout, layers=6, skip=1, p=6,
                 scale=10**6, pattern=[]):
        """Construct and return non-dimensionalized geometry."""

        geo = geom2d.SplineGeometry()

        # Non-Dimensionalize needed physical parameters

        sep /= scale
        r /= scale

        # Create core region

        if skip == 0:
            pass

        else:
            R = .5 * (sep * skip - r)
            geo.AddCircle(c=(0, 0), r=R, leftdomain=1, rightdomain=2)

        for i in range(layers):
            index_layer = skip + i
            R = index_layer * sep

            if len(pattern) > 0:
                self.add_layer(geo, r, R, p=p, innerpoints=index_layer - 1,
                               mask=pattern[i])
            else:
                self.add_layer(geo, r, R, p=p, innerpoints=index_layer - 1)

        # create boundary of fiber

        if R0 == R_air:
            geo.AddCircle(c=(0, 0), r=R0, leftdomain=2, rightdomain=5)
            geo.AddCircle(c=(0, 0), r=Rout, leftdomain=5,
                          bc="OuterCircle")  # outermost circle

        else:
            geo.AddCircle(c=(0, 0), r=R0, leftdomain=2, rightdomain=4)
            geo.AddCircle(c=(0, 0), r=R_air, leftdomain=4,
                          rightdomain=5)    # air/pml boundary
            geo.AddCircle(c=(0, 0), r=Rout, leftdomain=5,
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
            geo.AddCircle(c=(0, 0), r=r, leftdomain=3, rightdomain=2)

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
                geo.AddCircle(c=(x, y), r=r, leftdomain=3, rightdomain=2)

    # SAVE & LOAD #####################################################

    def save(self, fileprefix):
        """ Save this object so it can be loaded later """

        pbgfilename = os.path.relpath(self.outfolder+'/'+fileprefix+'_pbg.pkl')
        if os.path.isdir(self.outfolder):

            # os.makedirs(os.path.dirname(pbgfilename), exist_ok=True)
            print('Pickling pbg object into ', pbgfilename)
            with open(pbgfilename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise OSError("Your current objects outfolder is not a directory.\
                          Please check it and reset or make directory as nec\
                              cessary.")

    def savemodes(self, fileprefix, Y, p, betas, Zs,
                  solverparams, longY=None, longYl=None, pbgpickle=False):
        """
        Save a NGVec span object Y containing modes of FE degree p.
        Include any solver paramaters to be saved together with the
        modes in the input dictionary "solverparams". If "pbgpickle"
        is True, then the pbg object is also save under the same "fileprefix".
        """

        if pbgpickle:
            self.save(fileprefix)
        y = Y.tonumpy()
        if longY is not None:
            longY = longY.tonumpy()
        if longYl is not None:
            longYl = longYl.tonumpy()
        d = {'y': y, 'p': p, 'betas': betas, 'Zs': Zs,
             'longy': longY, 'longyl': longYl}
        d.update(**solverparams)

        f = os.path.relpath(self.outfolder+'/'+fileprefix+'_mde.npz')
        print('Writing mode file ', f)
        np.savez(f, **d)

# LOAD FROM FILE #####################################################


def load_pbg(fileprefix):
    """ Load a saved pbg object from file <fileprefix>_pbg.pkl """

    pbgfile = os.path.relpath(fileprefix+'_pbg.pkl')
    with open(pbgfile, 'rb') as f:
        a = pickle.load(f)
    return a


def load_pbg_mode(modenpzf, pbgfprefix):
    """  Load a mode saved in npz file <modenpzf> compatible with the
    pbg object saved in pickle file <pbgfprefix>_pbg.pkl. """

    a = load_pbg(pbgfprefix)
    modef = os.path.relpath(modenpzf + '_mde.npz')
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
