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

from warnings import warn
from pyeigfeast.spectralproj.ngs import NGvecs
from fiberamp.fiber import ModeSolver


class Bragg(ModeSolver):
    """
    Create a Bragg type multilayered radial fiber.

    Bragg fibers consist of a circular core region surrounded by many
    concentric circular layers of alternating material, often glass and air.

    """

    def __init__(self, scale=5e-5, ts=[5e-5, 1e-5, 2e-5, 2e-5],
                 mats=['air', 'glass', 'air', 'Outer'], ns=[1, 1.44, 1, 1],
                 maxhs=[.2, .025, .08, .1], bcs=None, wl=1.2e-6, ref=0,
                 curve=8, fan=False, beta_sq_plane=False):

        # Check inputs for errors
        self.check_parameters(ts, ns, mats, maxhs, bcs)

        self.L = scale
        self.scale = scale
        self.ts = ts
        self.mats = mats

        if bcs is not None:
            self.bcs = bcs
        else:
            self.bcs = ['r'+str(i+1) for i in range(len(ts))]
            self.bcs[-1] = 'OuterCircle'
            self.bcs[-2] = 'R'

        self.Ts = np.array(ts) / scale
        self.Rs = [sum(self.Ts[:i]) for i in range(1, len(self.Ts)+1)]
        self.maxhs = np.array(maxhs) * self.Rs

        self.R, self.Rout = self.Rs[-2:]
        self.wavelength = wl
        self.ns = ns

        # Create geometry
        self.create_geometry(fan=fan)
        self.create_mesh(ref=ref, curve=curve)
        self.set_material_properties(beta_sq_plane)

        super(Bragg, self).__init__(self.mesh, self.L, self.n0)

    @classmethod
    def from_dict(cls, d):
        """
        Create Bragg object from dictionary.
        """
        new_dict = {}
        list_of_keys = ['scale', 'ts', 'mats', 'ns', 'maxhs', 'bcs', 'wl',
                        'ref', 'curve', 'fan', 'beta_sq_plane']
        for key in list_of_keys:
            if key in d:
                new_dict[key] = d[key]
        return cls(**new_dict)

    def check_parameters(self, ts, ns, mats, maxhs, bcs):

        # Check that all relevant inputs have same length
        lengths = [len(ts), len(ns), len(mats), len(maxhs)]
        names = ['ts', 'ns', 'mats', 'maxhs']
        if bcs is not None:
            if bcs[-1] != 'OuterCircle':
                raise ValueError('Final boundary must be "OuterCircle".')
            lengths.append(len(bcs))
            names.append('bcs')
        else:
            print('Boundary names not provided, using default names.')
        lengths = np.array(lengths)

        same = all(x == lengths[0] for x in lengths)

        if not same:
            string = "Provided parameters not of same length: \n\n"
            for name, length in zip(names, lengths):
                string += name + ': ' + str(length) + '\n'
            raise ValueError(string + "\nModify above inputs as necessary and \
try again.")

        if mats[-1] != 'Outer':
            raise ValueError('Final material for PML region must be called\
"Outer".')

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

    def create_geometry(self, fan=False):
        """Construct and return Non-Dimensionalized geometry."""
        self.geo = geom2d.SplineGeometry()

        if not fan:
            for i, R in enumerate(self.Rs[:-1]):
                self.geo.AddCircle(c=(0, 0), r=R, leftdomain=i+1,
                                   rightdomain=i+2, bc=self.bcs[i])

            self.geo.AddCircle(c=(0, 0), r=self.Rs[-1],
                               leftdomain=len(self.Rs),
                               bc=self.bcs[-1])

            for i, (mat, maxh) in enumerate(zip(self.mats, self.maxhs)):
                self.geo.SetMaterial(i+1, mat)
                self.geo.SetDomainMaxH(i+1, maxh)
        else:
            for i, R in enumerate(self.Rs[:-3]):
                self.geo.AddCircle(c=(0, 0), r=R, leftdomain=i+1,
                                   rightdomain=i+2, bc=self.bcs[i])

            self.add_fan()

            for i, (mat, maxh) in enumerate(zip(self.mats, self.maxhs)):
                self.geo.SetMaterial(i+1, mat)
                self.geo.SetDomainMaxH(i+1, maxh)

    def set_material_properties(self, beta_sq_plane=False):
        """
        Set k0, refractive indices, and V function.
        """
        if beta_sq_plane:
            warn('Using square beta plane: search centers should be at\
 -(beta*L)**2 where L is scale attribute of fiber.')
        self.k = 2 * np.pi / self.wavelength
        self.refractive_indices = [self.ns[i](self.wavelength)
                                   if callable(self.ns[i])
                                   else self.ns[i]
                                   for i in range(len(self.ns))]
        self.index = ng.CF(self.refractive_indices)
        self.n0 = self.refractive_indices[-1]

        n0sq = ng.CF([self.n0**2 for i in range(len(self.ns))])
        self.V = (self.L * self.k)**2 * (n0sq * (not beta_sq_plane)
                                         - self.index ** 2)

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

    # Add fan to outer geometry #####################################

    def add_fan(self):
        r1, r2, r3 = self.Rs[-3:]

        dom_indx = [i+1 for i in range(len(self.Rs))]
        dom_indx = dom_indx[-3:]

        lower_points = [(-r1, 0), (-r1, -r1), (0, -r1), (r1, -r1), (r1, 0)]
        inner_points = [(r1, r1), (0, r1), (-r1, r1)]

        mid_points = [(r2, 0), (r2, r2), (0, r2), (-r2, r2), (-r2, 0)]
        outer_points = [(r3, 0), (r3, r3), (0, r3), (-r3, r3), (-r3, 0)]

        lower_pt_ids = [self.geo.AppendPoint(*p) for p in lower_points]

        inner_pt_ids = [self.geo.AppendPoint(*p) for p in inner_points]
        inner_pt_ids.insert(0, lower_pt_ids[-1])
        inner_pt_ids.append(lower_pt_ids[0])

        mid_pt_ids = [self.geo.AppendPoint(*p) for p in mid_points]

        outer_pt_ids = [self.geo.AppendPoint(*p) for p in outer_points]

        seg_pts = [outer_pt_ids[-1], mid_pt_ids[-1], lower_pt_ids[0],
                   lower_pt_ids[-1], mid_pt_ids[0], outer_pt_ids[0]]

        NP = len(lower_pt_ids)

        for k in range(0, NP - 1, 2):
            bc = 'OuterCircle'
            self.geo.Append(
                ['spline3',
                 lower_pt_ids[k % NP],
                 lower_pt_ids[(k + 1) % NP],
                 lower_pt_ids[(k + 2) % NP]], leftdomain=dom_indx[-3], bc=bc)

        NP = len(inner_pt_ids)

        for k in range(0, NP - 1, 2):
            bc = 'fiber_buffer_interface'
            self.geo.Append(
                ['spline3',
                 inner_pt_ids[k % NP],
                 inner_pt_ids[(k + 1) % NP],
                 inner_pt_ids[(k + 2) % NP]],
                leftdomain=dom_indx[-3],
                rightdomain=dom_indx[-2],
                bc=bc)

        NP = len(mid_pt_ids)

        for k in range(0, NP-1, 2):
            bc = 'buffer_Outer_interface'
            self.geo.Append(
                ['spline3',
                 mid_pt_ids[k % NP],
                 mid_pt_ids[(k + 1) % NP],
                 mid_pt_ids[(k + 2) % NP]],
                leftdomain=dom_indx[-2],
                rightdomain=dom_indx[-1],
                bc=bc)

        NP = len(outer_pt_ids)

        for k in range(0, NP-1, 2):
            bc = 'OuterCircle'
            self.geo.Append(
                ['spline3',
                 outer_pt_ids[k % NP],
                 outer_pt_ids[(k + 1) % NP],
                 outer_pt_ids[(k + 2) % NP]],
                leftdomain=dom_indx[-1],
                bc=bc)

        self.geo.Append(["line", seg_pts[0], seg_pts[1]], bc='OuterCircle',
                        leftdomain=dom_indx[-1])

        self.geo.Append(["line", seg_pts[1], seg_pts[2]], bc='OuterCircle',
                        leftdomain=dom_indx[-2])

        self.geo.Append(["line", seg_pts[3], seg_pts[4]], bc='OuterCircle',
                        leftdomain=dom_indx[-2])

        self.geo.Append(["line", seg_pts[4], seg_pts[5]], bc='OuterCircle',
                        leftdomain=dom_indx[-1])
