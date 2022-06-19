#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:04 2022

@author: pv
"""

import ngsolve as ng
import numpy as np
import pickle
from netgen.geom2d import CSG2d, Circle, Solid2d
from fiberamp.fiber.modesolver import ModeSolver
from pyeigfeast.spectralproj.ngs.spectralprojngs import NGvecs


class ARF2(ModeSolver):
    """
    Create an ARF fiber using csg2d
    """

    def __init__(self, name=None, refine=0, curve=3, e=None,
                 poly_core=False, shift_capillaries=False):

        # Set and check the fiber parameters.
        self.set_parameters(name=name, shift_capillaries=shift_capillaries,
                            e=e)
        self.check_parameters()

        # Create geometry
        self.create_geometry(poly_core=poly_core)
        self.create_mesh(refine=refine, curve=curve)

        # Set physical properties
        self.set_material_properties()

        super().__init__(self.mesh, self.scale, self.n0)

    def set_parameters(self, name=None, e=None, shift_capillaries=False):
        """
        Set fiber parameters.
        """
        # Default is 6-capillary fiber.
        self.name = 'poletti' if name is None else name

        if self.name == 'poletti':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.48 / scaling
            self.T_tube = .42 / scaling

            self.T_sheath = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling

            if shift_capillaries:
                self.R_sheath = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                 self.T_tube)

                self.R_tube_center = (self.R_sheath - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_sheath = (1 + 2 * self.R_tube + (2 - self.e) *
                                 self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.inner_air_maxh = .44
            self.fill_air_maxh = .44
            self.tube_maxh = .11
            self.sheath_maxh = .3
            self.buffer_maxh = 2
            self.outer_maxh = 4
            self.core_maxh = .25
            self.glass_maxh = 0

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_buffer = 1.00027717
            self.n0 = 1.00027717

            self.wavelength = 1.8e-6

        elif self.name == 'basic':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.48 / scaling
            self.T_tube = .84 / scaling

            self.T_sheath = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling

            if shift_capillaries:
                self.R_sheath = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                 self.T_tube)

                self.R_tube_center = (self.R_sheath - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_sheath = (1 + 2 * self.R_tube + (2 - self.e) *
                                 self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.inner_air_maxh = .44
            self.fill_air_maxh = .44
            self.tube_maxh = .11
            self.sheath_maxh = .3
            self.buffer_maxh = 2
            self.outer_maxh = 4
            self.core_maxh = .25
            self.glass_maxh = 0

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_buffer = 1.00027717
            self.n0 = 1.00027717

            self.wavelength = 1.8e-6

        else:
            err_str = 'Fiber \'{:s}\' not implemented.'.format(self.name)
            raise NotImplementedError(err_str)

    def check_parameters(self):
        """Check to ensure given parameters give valid geometry."""
        if self.e <= 0 or self.e >= 1:
            raise ValueError('Embedding parameter e must be strictly between\
 zero and one.')

        if self.n_tubes >= 2:
            if self.R_tube_center * np.sin(np.pi / self.n_tubes) \
                    <= self.R_tube + self.T_tube:
                raise ValueError('Capillary tubes overlap each other.')

    def set_material_properties(self):
        """
        Set k0, refractive indices, and V function.
        """
        self.k = 2 * np.pi / self.wavelength

        refractive_index_dict = {'core': self.n_air,
                                 'fill_air': self.n_air,
                                 'glass': self.n_glass,
                                 'inner_air': self.n_air,
                                 'buffer': self.n_buffer,
                                 'Outer': self.n0
                                 }
        self.refractive_index_dict = refractive_index_dict

        self.index = self.mesh.RegionCF(ng.VOL, refractive_index_dict)

        self.V = (self.scale * self.k)**2 * (self.n0 ** 2 - self.index ** 2)

    def create_mesh(self, refine=0, curve=3):
        """
        Create mesh from geometry.
        """
        self.mesh = ng.Mesh(self.geo.GenerateMesh())

        self.refinements = 0

        for i in range(refine):
            self.mesh.ngmesh.Refine()

        self.refinements += refine

        self.mesh.ngmesh.SetGeometry(self.spline_geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())

        self.mesh.Curve(curve)

    def create_geometry(self, poly_core=True):
        """
        Create geometry and mesh
        """

        geo = CSG2d()

        n_tubes = self.n_tubes
        R_core = self.R_core
        R_sheath = self.R_sheath
        T_sheath = self.T_sheath
        T_buffer = self.T_buffer
        T_outer = self.T_outer
        R_tube = self.R_tube
        T_tube = self.T_tube
        R_tube_center = self.R_tube_center

        self.R = R_sheath + T_sheath + T_buffer
        self.Rout = R_sheath + T_sheath + T_buffer + T_outer

        if self.n_tubes <= 2 and poly_core:
            raise ValueError('Polygonal core only available for n_tubes>2.')

        if poly_core:

            R_poly = self.R_core / np.cos(np.pi / n_tubes)
            core_points = [(-R_poly * np.sin((2 * i - 1) * np.pi / n_tubes),
                            R_poly * np.cos((2 * i - 1) * np.pi / n_tubes))
                           for i in range(n_tubes)]

            core = Solid2d(core_points, mat="core",
                           bc="core_fill_air_interface")
        else:
            core = Circle(center=(0, 0), radius=R_core,
                          mat="core", bc="core_fill_air_interface")

        circle1 = Circle(center=(0, 0),
                         radius=R_sheath,
                         mat="fill_air", bc="glass_air_interface")
        circle2 = Circle(center=(0, 0),
                         radius=R_sheath+T_sheath,
                         mat="glass", bc="sheathing_buffer_interface")
        circle3 = Circle(center=(0, 0),
                         radius=self.R,
                         mat="buffer", bc="buffer_pml_interface")
        circle4 = Circle(center=(0, 0),
                         radius=self.Rout,
                         mat="Outer", bc="OuterCircle")

        tube_maxh = self.tube_maxh
        sheath_maxh = self.sheath_maxh
        inner_air_maxh = self.inner_air_maxh
        fill_air_maxh = self.fill_air_maxh
        core_maxh = self.core_maxh
        buffer_maxh = self.buffer_maxh
        outer_maxh = self.outer_maxh
        glass_maxh = self.glass_maxh

        small1 = Circle(center=(0, R_tube_center), radius=R_tube,
                        mat="inner_air", bc="glass_air_interface")
        small2 = Circle(center=(0, R_tube_center), radius=R_tube+T_tube,
                        mat="glass", bc="glass_air_interface")

        inner_tubes = Solid2d()
        outer_tubes = Solid2d()

        for i in range(0, n_tubes):
            inner_tubes += inner_tubes + \
                small1.Copy().Rotate(360/n_tubes * i, center=(0, 0))
            outer_tubes += outer_tubes + \
                small2.Copy().Rotate(360/n_tubes * i, center=(0, 0))

        sheath = circle2 - circle1
        tubes = outer_tubes - inner_tubes

        tubes.Maxh(tube_maxh)
        sheath.Maxh(sheath_maxh)
        inner_tubes.Maxh(inner_air_maxh)

        glass = sheath + tubes
        glass.Mat('glass')

        if glass_maxh > 0:  # setting glass maxh overrides tube and sheath maxh
            glass.Maxh(glass_maxh)

        fill_air = circle1 - tubes - inner_tubes - core
        fill_air.Maxh(fill_air_maxh)
        fill_air.Mat('fill_air')

        core.Maxh(core_maxh)
        core.Mat('core')

        buffer_air = circle3 - circle2
        buffer_air.Maxh(buffer_maxh)
        buffer_air.Mat('buffer')

        outer = circle4 - circle3
        outer.Maxh(outer_maxh)
        outer.Mat('Outer')
        inner_tubes.Mat('inner_air')

        geo.Add(core)
        geo.Add(fill_air)
        geo.Add(glass)
        if n_tubes > 0:
            geo.Add(inner_tubes)
        geo.Add(buffer_air)
        geo.Add(outer)

        self.geo = geo
        self.spline_geo = geo.GenerateSplineGeometry()

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
        Y = ng.H1(self.mesh, order=p+1, dirichlet='OuterCircle', complex=True)
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
        with open(name, 'wb') as f:
            pickle.dump(self.mesh, f)

    def save_modes(self, modes, name):
        """Save modes as numpy arrays."""
        np.save(name, modes.tonumpy())

    def load_mesh(self, name):
        """ Load a saved ARF mesh."""
        with open(name, 'rb') as f:
            pmesh = pickle.load(f)
        return pmesh

    def load_E_modes(self, mesh_name, mode_name, p=8):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name)
        array = np.load(mode_name+'.npy')
        return self.E_modes_from_array(array, mesh=mesh, p=p)

    def load_phi_modes(self, mesh_name, mode_name, p=8):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name)
        array = np.load(mode_name+'.npy')
        return self.phi_modes_from_array(array, mesh=mesh, p=p)
