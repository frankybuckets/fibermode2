#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:04 2022

@author: pv
"""

import ngsolve as ng
import numpy as np
from netgen.geom2d import CSG2d, Circle, Solid2d
from fiberamp.fiber.modesolver import ModeSolver


class ARF2(ModeSolver):
    """
    Create an ARF fiber using csg2d
    """

    def __init__(self, name=None, refine=0, curve=3, e=None, poly_core=False):

        # Set and check the fiber parameters.
        self.set_parameters(name=name, e=e)
        self.check_parameters()

        # Create geometry
        self.create_geometry(poly_core=poly_core)
        self.create_mesh(refine=refine, curve=curve)

        # Set physical properties
        self.set_material_properties()

        super().__init__(self.mesh, self.scale, self.n0)

    def set_parameters(self, name=None, e=None):
        """
        Set fiber parameters.
        """
        # By default, we'll use the 6-capillary fiber.
        self.name = 'poletti' if name is None else name

        if self.name == 'poletti':
            # This case gives the default attributes of the fiber.
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

            self.R_sheath = (1 + 2 * self.R_tube + (2 - self.e) *
                             self.T_tube)

            self.R_tube_center = (self.R_sheath - self.R_tube -
                                  (1 - self.e) * self.T_tube)

            self.core_factor = .75
            self.R_core = ((self.R_tube_center - self.R_tube -
                           self.T_tube) * self.core_factor)

            self.inner_air_maxh = 4
            self.fill_air_maxh = .35
            self.tube_maxh = .22
            self.sheath_maxh = .5
            self.buffer_maxh = 2
            self.outer_maxh = 4
            self.core_maxh = .25

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_buffer = 1.00027717
            self.n0 = 1.00027717

            self.wavelength = 1.8e-6

        elif self.name == 'basic':
            # This case gives initial trial parameters.
            self.n_tubes = 6
            self.R_sheath = 2
            self.T_sheath = 1.5

            self.R_tube = .5
            self.T_tube = .1

            self.T_outer = 2
            self.T_buffer = 1.5

            if e is not None:
                self.e = e
            else:
                self.e = .5

            self.R_tube_center = self.R_sheath - self.R_tube - \
                (1 - self.e) * self.T_tube

            self.core_factor = .85
            self.R_core = (self.R_tube_center - self.R_tube -
                           self.T_tube) * self.core_factor

            self.inner_air_maxh = .3
            self.fill_air_maxh = .6
            self.tube_maxh = .1
            self.sheath_maxh = 1
            self.buffer_maxh = 2
            self.outer_maxh = 3
            self.core_maxh = .2

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_buffer = 1.00027717
            self.n0 = 1.00027717
            self.scale = 1e-6

            self.wavelength = 1.8e-6

        else:
            err_str = 'Fiber \'{:s}\' not implemented.'.format(self.name)
            raise NotImplementedError(err_str)

    def check_parameters(self):
        """Check to ensure given parameters give valid geometry."""
        pass

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
        # mats = refractive_index_dict.keys()
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

        if n_tubes <= 2 and poly_core:

            print('Polygonal core only available for n_tubes>2, setting round\
 core.')
            poly_core = False

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

        if n_tubes == 0:

            fill_air = circle1 - core
            glass = circle2 - circle1

            glass.Maxh(sheath_maxh)
            glass.Mat('glass')

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

            geo.Add(core)
            geo.Add(fill_air)
            geo.Add(glass)
            geo.Add(buffer_air)
            geo.Add(outer)

            self.geo = geo
            self.spline_geo = geo.GenerateSplineGeometry()

        else:

            small1 = Circle(center=(0, R_tube_center), radius=R_tube,
                            mat="inner_air", bc="glass_air_interface")
            small2 = Circle(center=(0, R_tube_center), radius=R_tube+T_tube,
                            mat="glass", bc="glass_air_interface")

            inner_tubes = small1.Copy()
            outer_tubes = small2.Copy()

            for i in range(1, n_tubes):
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
            geo.Add(inner_tubes)
            geo.Add(buffer_air)
            geo.Add(outer)

            self.geo = geo
            self.spline_geo = geo.GenerateSplineGeometry()
