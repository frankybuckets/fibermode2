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
from netgen.geom2d import EdgeInfo as EI
from fiberamp.fiber.modesolver import ModeSolver
from pyeigfeast.spectralproj.ngs.spectralprojngs import NGvecs


class ARF2(ModeSolver):
    """
    Create an ARF fiber using csg2d
    """

    def __init__(self, name=None, refine=0, curve=3, e=None,
                 poly_core=False, shift_capillaries=False,
                 outer_materials=None, fill=None):

        # Set and check the fiber parameters.
        self.set_parameters(name=name, shift_capillaries=shift_capillaries,
                            e=e, outer_materials=outer_materials)
        self.check_parameters()

        if self.name == 'original':  # Override for original fiber
            poly_core = False

        # Create geometry
        self.create_geometry(poly_core=poly_core, fill=fill)
        self.create_mesh(refine=refine, curve=curve)

        # Set physical properties
        self.set_material_properties()

        super().__init__(self.mesh, self.scale, self.n0)

    def set_parameters(self, name=None, e=None, shift_capillaries=False,
                       outer_materials=None):
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

            self.T_cladding = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling
            self.T_soft_polymer = 30 / scaling
            self.T_hard_polymer = 30 / scaling

            if shift_capillaries:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                   self.T_tube)

                self.R_tube_center = (self.R_cladding - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - self.e) *
                                   self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_soft_polymer = 1.44
            self.n_hard_polymer = 1.56
            self.n_buffer = self.n_air
            self.n0 = self.n_air

            if outer_materials is not None:
                self.outer_materials = outer_materials
            else:
                self.outer_materials = [

                    # {'material': 'soft_polymer',
                    #  'n': self.n_soft_polymer,
                    #  'T': self.T_soft_polymer,
                    #  'maxh': 2},

                    # {'material': 'hard_polymer',
                    #  'n': self.n_hard_polymer,
                    #  'T': self.T_hard_polymer,
                    #  'maxh': 2},

                    {'material': 'buffer',
                     'n': self.n_buffer,
                     'T': self.T_buffer,
                     'maxh': 2},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 2}
                ]

            self.inner_air_maxh = .2
            self.fill_air_maxh = .35
            self.tube_maxh = .11
            self.cladding_maxh = .25
            self.core_maxh = .25
            self.glass_maxh = 0  # Overrides maxh in tubes and cladding

            self.wavelength = 1.8e-6

        elif self.name == 'original':  # Ben's original parameters

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.48 / scaling
            self.T_tube = .42 / scaling

            self.T_cladding = 10 / scaling
            self.T_outer = 50 / scaling
            self.T_buffer = 10 / scaling
            self.T_soft_polymer = 30 / scaling
            self.T_hard_polymer = 30 / scaling

            self.R_cladding = (1 + 2 * self.R_tube + (2 - self.e) *
                               self.T_tube)

            self.R_tube_center = 1 + self.R_tube + self.T_tube
            self.core_factor = .75
            self.R_core = self.core_factor

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_soft_polymer = 1.44
            self.n_hard_polymer = 1.56
            self.n_buffer = self.n_air
            self.n0 = self.n_air

            if outer_materials is not None:
                self.outer_materials = outer_materials
            else:
                self.outer_materials = [

                    # {'material': 'soft_polymer',
                    #  'n': self.n_soft_polymer,
                    #  'T': self.T_soft_polymer,
                    #  'maxh': 2},

                    # {'material': 'hard_polymer',
                    #  'n': self.n_hard_polymer,
                    #  'T': self.T_hard_polymer,
                    #  'maxh': 2},

                    {'material': 'buffer',
                     'n': self.n_buffer,
                     'T': self.T_buffer,
                     'maxh': 2},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 4}
                ]

            self.inner_air_maxh = .25
            self.fill_air_maxh = .25
            self.tube_maxh = .04
            self.cladding_maxh = .33
            self.core_maxh = .1
            self.glass_maxh = 0  # Overrides maxh in tubes and cladding

            self.wavelength = 1.8e-6

        elif self.name == 'basic':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.06 / scaling
            self.T_tube = .84 / scaling

            self.T_cladding = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling
            self.T_soft_polymer = 30 / scaling
            self.T_hard_polymer = 30 / scaling

            if shift_capillaries:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                   self.T_tube)

                self.R_tube_center = (self.R_cladding - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - self.e) *
                                   self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_soft_polymer = 1.44
            self.n_hard_polymer = 1.56
            self.n_buffer = self.n_air
            self.n0 = self.n_air

            if outer_materials is not None:
                self.outer_materials = outer_materials
            else:
                self.outer_materials = [

                    # {'material': 'soft_polymer',
                    #  'n': self.n_soft_polymer,
                    #  'T': self.T_soft_polymer,
                    #  'maxh': 2},

                    # {'material': 'hard_polymer',
                    #  'n': self.n_hard_polymer,
                    #  'T': self.T_hard_polymer,
                    #  'maxh': 2},

                    {'material': 'buffer',
                     'n': self.n_buffer,
                     'T': self.T_buffer,
                     'maxh': 2},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 4}
                ]

            self.core_maxh = .25
            self.fill_air_maxh = .35
            self.tube_maxh = .11
            self.inner_air_maxh = .2
            self.cladding_maxh = .25
            self.glass_maxh = 0  # Overrides maxh in tubes and cladding

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

        # Set up inner region refractive indices
        refractive_index_dict = {'core': self.n_air,
                                 'fill_air': self.n_air,
                                 'glass': self.n_glass,
                                 'inner_air': self.n_air,
                                 }

        # Add outer material refractive indices
        for i, d in enumerate(self.outer_materials):
            refractive_index_dict[d['material']] = d['n']

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

    def create_geometry(self, poly_core=True, fill=None):
        """
        Create geometry and mesh
        """

        geo = CSG2d()

        n_tubes = self.n_tubes
        R_core = self.R_core
        R_cladding = self.R_cladding
        T_cladding = self.T_cladding
        R_tube = self.R_tube
        T_tube = self.T_tube
        R_tube_center = self.R_tube_center

        if self.n_tubes <= 2 and poly_core:
            raise ValueError('Polygonal core only available for n_tubes > 2.')

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
                         radius=R_cladding,
                         mat="fill_air", bc="glass_air_interface")
        circle2 = Circle(center=(0, 0),
                         radius=R_cladding+T_cladding,
                         mat="glass", bc="cladding_outer_materials_interface")

        small1 = Circle(center=(0, R_tube_center), radius=R_tube,
                        mat="inner_air", bc="glass_air_interface")
        small2 = Circle(center=(0, R_tube_center), radius=R_tube+T_tube,
                        mat="glass", bc="glass_air_interface")

        # previous method
        inner_tubes = small1.Copy()
        outer_tubes = small2.Copy()

        for i in range(1, n_tubes):
            inner_tubes += small1.Copy().Rotate(360/n_tubes * i,
                                                center=(0, 0))
            outer_tubes += small2.Copy().Rotate(360/n_tubes * i,
                                                center=(0, 0))
        if fill is not None:
            fills = self.get_fill(fill)
            fills = fills - small2
            all_fill = fills.Copy()
            for i in range(1, n_tubes):
                all_fill += fills.Copy().Rotate(360/n_tubes * i,
                                                center=(0, 0))

        # # Second method
        # inner_tubes = Solid2d()
        # outer_tubes = Solid2d()

        # for i in range(0, n_tubes):
        #     inner_tubes += small1.Copy().Rotate(360/n_tubes * i,
        #                                         center=(0, 0))
        #     outer_tubes += small2.Copy().Rotate(360/n_tubes * i,
        #                                         center=(0, 0))

        cladding = circle2 - circle1
        tubes = outer_tubes - inner_tubes

        tubes.Maxh(self.tube_maxh)
        cladding.Maxh(self.cladding_maxh)
        inner_tubes.Maxh(self.inner_air_maxh)

        glass = cladding + tubes

        if fill is not None:
            glass += all_fill

        glass.Mat('glass')

        if self.glass_maxh > 0:  # setting overrides tube and cladding maxh
            glass.Maxh(self.glass_maxh)

        fill_air = circle1 - outer_tubes - core

        if fill is not None:
            fill_air = fill_air - all_fill

        fill_air.Maxh(self.fill_air_maxh)
        fill_air.Mat('fill_air')

        core.Maxh(self.core_maxh)
        core.Mat('core')

        geo.Add(core)
        geo.Add(fill_air)
        geo.Add(glass)

        if n_tubes > 0:  # Meshing fails if you add empty Solid2d instance
            geo.Add(inner_tubes)
            pass

        # Create and add outer materials including PML region

        self.Rout = R_cladding + T_cladding  # set base Rout

        n = len(self.outer_materials)

        for i, d in enumerate(self.outer_materials):

            circle1 = circle2  # increment circles

            self.Rout += d['T']  # add thickness of materials

            # Take care of boundary naming
            if i < (n-1):
                mat = d['material']
                next_mat = self.outer_materials[i+1]['material']
                bc = mat + '_' + next_mat + '_interface'
            else:
                bc = 'OuterCircle'
                mat = 'Outer'

            # Make new outermost circle
            circle2 = Circle(center=(0, 0),
                             radius=self.Rout,
                             mat=mat,
                             bc=bc)

            # Create and add region to geometry
            region = circle2 - circle1
            region.Mat(mat)
            region.Maxh(d['maxh'])
            geo.Add(region)

        self.R = self.Rout - self.outer_materials[-1]['T']
        self.geo = geo
        self.spline_geo = geo.GenerateSplineGeometry()

    def get_fill(self, fill):
        """Create fill."""
        # Get fill parameters from dictionary.
        beta, sigma = fill['beta'], fill['sigma']

        if beta <= 0 or beta > np.pi/6:
            raise ValueError('Fill angle beta must be in (0, pi/6]')

        if sigma < -1 or sigma > 1:
            raise ValueError('Fill convexity factor sigma must be in [-1,1]')

        # Get relevant names
        R_cladding = self.R_cladding
        T_cladding = self.T_cladding
        R_tube = self.R_tube
        T_tube = self.T_tube
        R_tube_center = self.R_tube_center

        # Point P: intersection of embedded capillary tube with cladding
        Py = (R_tube_center**2 + R_cladding**2 -
              (R_tube + T_tube)**2) / (2 * R_tube_center)
        Px = np.sqrt(R_cladding**2 - Py**2)

        # Point p: on radial line through capillary tube as P,
        # but shifted to lie inside capillary wall.
        # This assists with correct mesh construction for CSG2d.
        px, py = Px, Py - R_tube_center
        px *= (2*R_tube + T_tube)/(2*(R_tube + T_tube))
        py *= (2*R_tube + T_tube)/(2*(R_tube + T_tube))
        py += R_tube_center

        # Point Q: location on exterior of capillary tube at which fill begins
        Qx = Px * np.cos(beta) + (Py - R_tube_center) * np.sin(beta)
        Qy = (Py - R_tube_center) * np.cos(beta) - \
            Px * np.sin(beta) + R_tube_center

        # Point q: on radial line through capillary tube as Q,
        # but shifted to lie inside capillary wall.
        # This assists with correct mesh construction for CSG2d.
        qx, qy = Qx, Qy - R_tube_center
        qx *= (2*R_tube + T_tube)/(2*(R_tube + T_tube))
        qy *= (2*R_tube + T_tube)/(2*(R_tube + T_tube))
        qy += R_tube_center

        # Angle theta: angle away from vertical in which to go towards
        # cladding from point Q
        # This is chosen so that the angle with the capillary wall and
        # cladding wall are equal
        theta = np.arctan(Qx / (Qy - R_tube_center + (R_tube_center *
                          (R_tube + T_tube) / (R_cladding +
                                               (R_tube + T_tube)))))

        # Unit vector e: points from Q toward cladding in direction we want
        ex, ey = np.sin(theta), np.cos(theta)

        # Dot products
        Qe = Qx * ex + Qy * ey
        QQ = Qx * Qx + Qy * Qy

        # Length t: distance from outer capillary wall to cladding
        # along (flat) fill
        t = -Qe + np.sqrt(Qe**2 + R_cladding**2 - QQ)

        # Point T: Point on inner cladding wall where fill begins
        Tx, Ty = Qx + t * ex, Qy + t * ey

        # Vector TP
        TPx, TPy = Px - Tx, Py - Ty

        # Distance d: perpendicular distance from (flat) fill to line in
        # direction TP
        d = t/2 * np.sqrt((TPx * TPx + TPy * TPy)/(ex * TPx + ey * TPy)**2 - 1)

        # Modified points T and P lying inside interiors of relevant regions
        # Necessary to ensure fill extends to cladding and for meshing purposes
        O1x = (2*R_cladding + T_cladding) / (2*R_cladding) * Tx
        O1y = (2*R_cladding + T_cladding) / (2*R_cladding) * Ty

        O2x = (2*R_cladding + T_cladding) / (2*R_cladding) * Px
        O2y = (2*R_cladding + T_cladding) / (2*R_cladding) * Py

        # Control point c: lies along perpendicular to (flat) fill at
        # midpoint of fill.
        cx = Qx + t/2 * ex + d * sigma * ey
        cy = Qy + t/2 * ey - d * sigma * ex

        if sigma != 0:
            fill_r = Solid2d([(px, py), (qx, qy), (Qx, Qy), EI((cx, cy)),
                              (Tx, Ty), (O1x, O1y), (O2x, O2y)])
        else:
            fill_r = Solid2d([(px, py), (qx, qy), (Qx, Qy),
                              (Tx, Ty), (O1x, O1y), (O2x, O2y)])

        fill_l = fill_r.Copy().Scale((-1, 1))

        return fill_r + fill_l

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
