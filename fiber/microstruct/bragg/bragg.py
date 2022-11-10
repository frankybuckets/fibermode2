#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:48:03 2022

@author: pv

"""
import numpy as np
import netgen.geom2d as geom2d
import ngsolve as ng
import pickle

from pyeigfeast.spectralproj.ngs import NGvecs
from fiberamp.fiber import ModeSolver
from opticalmaterialspy import Air, SiO2


class Bragg(ModeSolver):
    """
    Create a Bragg type multilayered radial fiber.

    Bragg fibers consist of a circular core region surrounded by many
    concentric circular layers of alternating material, often glass and air.

    """

    def __init__(self, scale=1e-6, ts=[1e-6, .5e-6, .5e-6, .5e-6],
                 mats=['air', 'glass', 'air', 'Outer'],
                 ns=[Air().n, SiO2().n, Air().n, Air().n],
                 maxhs=[.4, .1, .1, .1],
                 wl=1.8e-6, ref=0, curve=8):

        # Check inputs for errors
        self.check_parameters(ts, ns, mats, maxhs)

        self.L = scale
        self.scale = scale
        self.ts = ts
        self.mats = mats
        self.Ts = np.array(ts) / scale
        self.Rs = [sum(self.Ts[:i]) for i in range(1, len(self.Ts)+1)]
        self.maxhs = np.array(maxhs) * self.Rs

        self.R, self.Rout = self.Rs[-2:]
        self.wavelength = wl
        self.ns = ns

        # Create geometry
        self.create_geometry()
        self.create_mesh(ref=ref, curve=curve)
        self.set_material_properties()

        super(Bragg, self).__init__(self.mesh, self.L, self.n0)

    def check_parameters(self, ts, ns, mats, maxhs):

        # Check that all relevant inputs have same length
        lengths = [len(ts), len(ns), len(mats), len(maxhs)]
        lengths = np.array(lengths)
        names = ['ts', 'ns', 'mats', 'maxhs']

        same = all(x == lengths[0] for x in lengths)

        if not same:
            string = "Provided parameters not of same length: \n\n"
            for name, length in zip(names, lengths):
                string += name + ': ' + str(length) + '\n'
            raise ValueError(string + "\nModify above inputs as necessary and \
try again.")

        all_callable = all(callable(ns[i]) for i in range(len(ns)))

        if not all_callable:
            raise ValueError("One of the provided ns is not callable.  \
Refractive indices in this class should be provided as callables to allow for \
dependence on wavelength.  If not desiring this dependence, provide fixed n \
as a lambda function: lambda x: n.")

    def create_mesh(self, ref=0, curve=8):
        """
        Create mesh from geometry.
        """
        self.mesh = ng.Mesh(self.geo.GenerateMesh())

        self.refinements = 0
        for i in range(ref):
            self.mesh.ngmesh.Refine()

        self.refinements += ref
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(curve)

    def create_geometry(self):
        """Construct and return Non-Dimensionalized geometry."""
        self.geo = geom2d.SplineGeometry()

        for i, R in enumerate(self.Rs[:-1]):
            self.geo.AddCircle(c=(0, 0), r=R, leftdomain=i+1,
                               rightdomain=i+2)

        self.geo.AddCircle(c=(0, 0), r=self.Rs[-1], leftdomain=len(self.Rs),
                           bc='OuterCircle')

        for i, (mat, maxh) in enumerate(zip(self.mats, self.maxhs)):
            self.geo.SetMaterial(i+1, mat)
            self.geo.SetDomainMaxH(i+1, maxh)

    def set_material_properties(self):
        """
        Set k0, refractive indices, and V function.
        """
        self.k = 2 * np.pi / self.wavelength
        self.refractive_indices = [self.ns[i](
            self.wavelength) for i in range(len(self.ns))]
        self.index = ng.CoefficientFunction(self.refractive_indices)
        self.n0 = self.refractive_indices[-1]

        self.V = (self.L * self.k)**2 * (self.n0 ** 2 - self.index ** 2)

    def E_modes_from_array(self, array, p=1, mesh=None):
        """Create NGvec object containing modes and set data given by array."""
        if mesh is None:
            mesh = self.mesh
        X = ng.HCurl(mesh, order=p+1-max(1-p, 0), type1=True,
                     dirichlet='OuterCircle', complex=True)
        m = array.shape[1]
        E = NGvecs(X, m)
        try:
            E.fromnumpy(array)
        except ValueError:
            raise ValueError("Array is wrong length: make sure your mesh is\
 constructed the same as for input array and that polynomial degree correspond\
ing to array has been passed as keyword p.")
        return E

    def phi_modes_from_array(self, array, p=1, mesh=None):
        """Create NGvec object containing modes and set data given by array."""
        if mesh is None:
            mesh = self.mesh
        Y = ng.H1(mesh, order=p+1, dirichlet='OuterCircle', complex=True)
        m = array.shape[1]
        phi = NGvecs(Y, m)
        try:
            phi.fromnumpy(array)
        except ValueError:
            raise ValueError("Array is wrong length: make sure your mesh is\
 constructed the same as for input array and that polynomial degree correspond\
ing to array has been passed as keyword p.")
        return phi

    # SAVE & LOAD #####################################################

    def save_mesh(self, name):
        """ Save mesh using pickle (allows for mesh curvature). """
        if name[-4:] != '.pkl':
            name += '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(self.mesh, f)

    def save_modes(self, modes, name):
        """Save modes as numpy arrays."""
        if name[-4:] == '.npy':
            name -= '.npy'
        np.save(name, modes.tonumpy())

    def load_mesh(self, name):
        """ Load a saved ARF mesh."""
        if name[-4:] != '.pkl':
            name += '.pkl'
        with open(name, 'rb') as f:
            pmesh = pickle.load(f)
        return pmesh

    def load_E_modes(self, mesh_name, mode_name, p=8):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name)
        if mode_name[-4:] == '.npy':
            mode_name -= '.npy'
        array = np.load(mode_name+'.npy')
        return self.E_modes_from_array(array, mesh=mesh, p=p)

    def load_phi_modes(self, mesh_name, mode_name, p=8):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name)
        array = np.load(mode_name+'.npy')
        return self.phi_modes_from_array(array, mesh=mesh, p=p)
