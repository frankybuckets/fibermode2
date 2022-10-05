#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:04 2022

@author: pv
"""

import ngsolve as ng
import numpy as np
import pickle
from netgen.geom2d import SplineGeometry
from fiberamp.fiber.modesolver import ModeSolver
from pyeigfeast.spectralproj.ngs.spectralprojngs import NGvecs


class ARF2(ModeSolver):
    """
    Create ARF fiber using spline geometry (again), which handles fill better.
    """

    def __init__(self, name=None, refine=0, curve=3, e=None,
                 poly_core=False, shift_capillaries=False,
                 outer_materials=None, fill=None, T_cladding=10,
                 wl=1.8e-6):

        # Set and check the fiber parameters.
        self.set_parameters(name=name, shift_capillaries=shift_capillaries,
                            e=e, outer_materials=outer_materials,
                            T_cladding=T_cladding, wl=wl)
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
                       outer_materials=None, T_cladding=10, wl=1.8e-6):
        """
        Set fiber parameters.
        """
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

            self.T_cladding = T_cladding / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling

            # self.T_soft_polymer = 30 / scaling
            # self.T_hard_polymer = 30 / scaling

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

            # self.n_soft_polymer = 1.44
            # self.n_hard_polymer = 1.56

            self.n_buffer = self.n_air
            self.n0 = self.n_air

            self.inner_air_maxh = .2
            self.fill_air_maxh = .35
            self.tube_maxh = .11
            self.cladding_maxh = .25

            self.inner_tube_edge_maxh = .11
            self.outer_tube_edge_maxh = .11
            self.inner_cladding_edge_maxh = .25
            self.outer_cladding_edge_maxh = .25
            self.fill_edge_maxh = .11

            self.core_maxh = .25
            self.glass_maxh = 0

            if outer_materials is not None:
                self.outer_materials = outer_materials
                self.n0 = outer_materials[-1]['n']  # Need to reset n0
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
                     'maxh': .5},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 4}
                ]

            self.wavelength = wl

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

            self.T_cladding = T_cladding / scaling
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

            self.inner_air_maxh = .2
            self.fill_air_maxh = .35
            self.tube_maxh = .11
            self.cladding_maxh = .25

            self.inner_tube_edge_maxh = .11
            self.outer_tube_edge_maxh = .11
            self.inner_cladding_edge_maxh = .25
            self.outer_cladding_edge_maxh = .25
            self.fill_edge_maxh = .11

            self.core_maxh = .25
            self.glass_maxh = 0

            if outer_materials is not None:
                self.outer_materials = outer_materials
                self.n0 = outer_materials[-1]['n']  # Need to reset n0
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

            self.wavelength = wl

        elif self.name == 'fine_cladding':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.48 / scaling
            self.T_tube = .42 / scaling

            self.T_cladding = T_cladding / scaling
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
                self.n0 = outer_materials[-1]['n']  # Need to reset n0
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
                     'maxh': .1},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 2}
                ]

            self.inner_air_maxh = .2
            self.fill_air_maxh = .35
            self.tube_maxh = .11
            self.cladding_maxh = .25

            self.inner_tube_edge_maxh = .11
            self.outer_tube_edge_maxh = .11
            self.inner_cladding_edge_maxh = .25
            self.outer_cladding_edge_maxh = .25
            self.fill_edge_maxh = .11

            self.core_maxh = .25
            self.glass_maxh = 0.05

            self.wavelength = wl

        else:
            err_str = 'Fiber \'{:s}\' not implemented.'.format(self.name)
            raise NotImplementedError(err_str)

    def check_parameters(self):
        """Check to ensure given parameters give valid geometry."""
        if self.e <= 0 or self.e > 1:
            raise ValueError('Embedding parameter e must be strictly greater\
 than zero and less than or equal to one.')

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
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(curve)

    def create_geometry(self, fill=None, poly_core=False):

        geo = SplineGeometry()

        n_tubes = self.n_tubes
        if n_tubes <= 2 and poly_core:
            print('Polygonal core only available for n_tubes > 2.')
            poly_core = False

        # We build the fiber starting from inside to outside: core first

        if poly_core:

            R_poly = self.R_core / np.cos(np.pi / n_tubes)

            core_points = [(-R_poly * np.sin((2 * i - 1) * np.pi / n_tubes),
                            R_poly * np.cos((2 * i - 1) * np.pi / n_tubes))
                           for i in range(n_tubes)]

            pts = [geo.AppendPoint(x, y) for x, y in core_points]

            for i in range(n_tubes - 1):
                geo.Append(["line", pts[i], pts[i+1]], leftdomain=1,
                           rightdomain=2, bc='core_fill_air_interface')

            geo.Append(["line", pts[-1], pts[0]], leftdomain=1,
                       rightdomain=2, bc='core_fill_air_interface')

        else:
            geo.AddCircle(c=(0, 0), r=self.R_core, leftdomain=1,
                          rightdomain=2, bc='core_fill_air_interface')

        # Now we add the microtubes and cladding

        if n_tubes == 0:
            geo.AddCircle(c=(0, 0), r=self.R_cladding, leftdomain=2,
                          rightdomain=4, bc='fill_air_cladding_interface')
        else:
            # The angle 'phi' corresponds to the polar angle that gives the
            # intersection of the two circles of radius Rcladi and Rto, resp.
            # The coordinates of the intersection can then be recovered as
            # (Rcladi * cos(phi), Rcladi * sin(phi)) and
            # (-Rcladi * cos(phi), Rcladi * sin(phi)).

            numerator = self.R_cladding**2 + \
                self.R_tube_center**2 - (self.R_tube + self.T_tube)**2
            denominator = 2 * self.R_tube_center * self.R_cladding
            acos_frac = numerator / denominator
            phi = np.arccos(acos_frac)

            # The angle of a given sector that bisects two adjacent capillary
            # tubes. Visually, this looks like a wedge in the computational
            # domain that contains a half capillary tube on each side of the
            # widest part of the wedge.
            sector = 2 * np.pi / self.n_tubes

            # Obtain the angle of the arc between two capillaries. This
            # subtends the arc between the two points where two adjacent
            # capillary tubes embed into the outer glass cladding.
            if fill is None:
                psi = sector - 2 * phi
                # Get the distance to the middle control point for the
                # aforementioned arc.
                D = self.R_cladding / np.cos(psi / 2)
            else:
                # Need to calculate polar angle nu from y axis to
                # fill/cladding interface
                _, _, _, nu = self.get_fill_points(
                    fill['delta'], fill['sigma'])
                psi = sector - 2 * nu
                D = self.R_cladding / np.cos(psi / 2)

            # The center of the top capillary tube.
            c = (0, self.R_tube_center)

            capillary_points = []

            if self.n_tubes == 1:
                rotation_angle = 0

                if fill is None:
                    hard1 = (self.R_cladding, self.R_cladding *
                             (1 - np.sin(phi)) / np.cos(phi))
                    hard2 = (-self.R_cladding, self.R_cladding *
                             (1 - np.sin(phi)) / np.cos(phi))
                else:
                    hard1 = (self.R_cladding, self.R_cladding *
                             (1 - np.sin(nu)) / np.cos(nu))
                    hard2 = (-self.R_cladding, self.R_cladding *
                             (1 - np.sin(nu)) / np.cos(nu))

                capillary_points += [hard1]

                capillary_points += \
                    self.get_capillary_spline_points(c, phi, rotation_angle,
                                                     fill=fill)

                capillary_points += [hard2]
                capillary_points += [(-self.R_cladding, 0),
                                     (-self.R_cladding, -self.R_cladding),
                                     (0, -self.R_cladding),
                                     (self.R_cladding, -self.R_cladding),
                                     (self.R_cladding, 0)]

            else:
                for k in range(self.n_tubes):
                    # Compute the rotation angle needed for rotating the north
                    # capillary spline points to the other capillary locations
                    # inthe domain.
                    rotation_angle = k * sector

                    # Compute the middle control point for the outer arc
                    # subtended by the angle psi + rotation_angle.
                    if fill is None:
                        ctrl_pt_angle = np.pi / 2 - phi - psi / 2 \
                            + rotation_angle
                    else:
                        ctrl_pt_angle = np.pi / 2 - nu - psi / 2 \
                            + rotation_angle

                    capillary_points += [(D * np.cos(ctrl_pt_angle),
                                          D * np.sin(ctrl_pt_angle))]

                    # Obtain the control points for the capillary tube
                    # immediately counterclockwise from the above control point
                    capillary_points += \
                        self.get_capillary_spline_points(
                            c, phi, rotation_angle, fill=fill)

            # Add the capillary points to the geometry
            capnums = [geo.AppendPoint(x, y) for x, y in capillary_points]
            NP = len(capillary_points)    # number of capillary point IDs.
            if fill is None:
                for k in range(1, NP + 1, 2):  # add the splines.
                    if k % 10 != 9:
                        maxh = self.outer_tube_edge_maxh
                        bc = 'fill_air_capillary_interface'
                    else:
                        maxh = self.inner_cladding_edge_maxh
                        bc = 'fill_air_cladding_interface'
                    geo.Append(
                        [
                            'spline3',
                            capnums[k % NP],
                            capnums[(k + 1) % NP],
                            capnums[(k + 2) % NP]
                        ], leftdomain=2, rightdomain=4, maxh=maxh,
                        bc=bc)
            else:
                for k in range(1, NP + 1, 2):  # add the splines.
                    if k % 14 == 1 or k % 14 == 11:
                        maxh = self.fill_edge_maxh
                        bc = 'fill_air_geometric_fill_interface'

                    elif k % 14 == 13:
                        maxh = self.inner_cladding_edge_maxh
                        bc = 'fill_air_cladding_interface'
                    else:
                        maxh = self.outer_tube_edge_maxh
                        bc = 'fill_air_capillary_interface'
                    geo.Append(
                        [
                            'spline3',
                            capnums[k % NP],
                            capnums[(k + 1) % NP],
                            capnums[(k + 2) % NP]
                        ], leftdomain=2, rightdomain=4, maxh=maxh,
                        bc=bc)

            # ----------------------------------------------------------------
            # Add capillary tubes.
            # ----------------------------------------------------------------

            # Spacing for the angles we need to add the inner circles for the
            # capillaries.
            theta = np.pi / 2.0 + np.linspace(0, 2*np.pi,
                                              num=self.n_tubes,
                                              endpoint=False)

            # The radial distance to the capillary tube centers.
            dist = self.R_tube_center

            for t in theta:
                c = (dist*np.cos(t), dist*np.sin(t))

                geo.AddCircle(c=c, r=self.R_tube,
                              leftdomain=3, rightdomain=4,
                              bc='inner_air_capillary_interface',
                              maxh=self.inner_tube_edge_maxh)

        self.Rout = self.R_cladding + self.T_cladding  # set base Rout

        geo.AddCircle(c=(0, 0), r=self.Rout, leftdomain=4, rightdomain=5,
                      bc='cladding_outer_materials_interface',
                      maxh=self.outer_cladding_edge_maxh)

        self.inner_materials = {
            'core': {'index': 1,
                     'maxh': self.core_maxh},
            'fill_air': {'index': 2,
                         'maxh': self.fill_air_maxh},
            'glass': {'index': 4,
                      'maxh': self.glass_maxh},
            'inner_air': {'index': 3,
                          'maxh': self.inner_air_maxh}
        }

        # Set inner material names and maxhs
        for material, info in self.inner_materials.items():
            geo.SetMaterial(info['index'], material)
            if material == 'glass' and info['maxh'] == 0:
                pass
            else:
                geo.SetDomainMaxH(info['index'], info['maxh'])

        # Create and add outer materials including PML region
        n = len(self.outer_materials)
        for i, d in enumerate(self.outer_materials):

            self.Rout += d['T']  # add thickness of materials

            # Take care of boundary naming
            if i < (n-1):
                mat = d['material']
                next_mat = self.outer_materials[i+1]['material']
                bc = mat + '_' + next_mat + '_interface'
                geo.AddCircle(c=(0, 0), r=self.Rout, leftdomain=(i+5),
                              rightdomain=(i+6), bc=bc)
                geo.SetMaterial(i+5, d['material'])
                geo.SetDomainMaxH(i+5, d['maxh'])
            else:
                bc = 'OuterCircle'
                geo.AddCircle(c=(0, 0), r=self.Rout, leftdomain=(i+5), bc=bc)
                geo.SetMaterial(i+5, d['material'])
                geo.SetDomainMaxH(i+5, d['maxh'])

        self.R = self.Rout - self.outer_materials[-1]['T']
        self.geo = geo

    def get_capillary_spline_points(self, c, phi, rotation_angle, fill=None):
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
        if fill is None:
            points = np.zeros((2, 9))

            # Compute the angle inside of the capillary tube to determine some
            # of the subsequent spline points for the upper half of the outer
            # capillary tube.
            acos_frac = (self.R_cladding * np.sin(phi) -
                         c[0]) / (self.R_tube + self.T_tube)
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

        else:
            points = np.zeros((2, 13))

            delta, sigma = fill['delta'], fill['sigma']
            T, C, Q, _ = self.get_fill_points(delta, sigma)

            # Compute the angle inside of the capillary tube to determine some
            # of the subsequent spline points for the upper half of the outer
            # capillary tube.
            acos_frac = (self.R_cladding * np.sin(phi) -
                         c[0]) / (self.R_tube + self.T_tube)
            psi = np.arccos(acos_frac) - delta

            # The control points for the right fill.
            points[:, 0] = [(T[0] - c[0])/(self.R_tube + self.T_tube),
                            (T[1] - c[1])/(self.R_tube + self.T_tube)]
            points[:, 1] = [(C[0] - c[0])/(self.R_tube + self.T_tube),
                            (C[1] - c[1])/(self.R_tube + self.T_tube)]

            # The control points for the first spline.
            points[:, 2] = [np.cos(psi), np.sin(psi)]
            points[:, 3] = [1, (1 - np.cos(psi)) / np.sin(psi)]
            points[:, 4] = [1, 0]

            # Control points for the second and third splines.
            points[:, 5] = [1, -1]
            points[:, 6] = [0, -1]
            points[:, 7] = [-1, -1]

            # Control points for the final spline.
            points[:, 8] = [-1, 0]
            points[:, 9] = [-1, (1 - np.cos(psi)) / np.sin(psi)]
            points[:, 10] = [-np.cos(psi), np.sin(psi)]

            # The control points for the left fill.
            points[:, 11] = [-(C[0] - c[0])/(self.R_tube + self.T_tube),
                             (C[1] - c[1])/(self.R_tube + self.T_tube)]
            points[:, 12] = [-(T[0] - c[0])/(self.R_tube + self.T_tube),
                             (T[1] - c[1])/(self.R_tube + self.T_tube)]

        # The rotation matrix needed to generate the spline points for an
        # arbitrary capillary tube.
        R = np.mat(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)]
            ]
        )

        # Rotate, scale, and shift the points.
        points *= (self.R_tube + self.T_tube)
        points[0, :] += c[0]
        points[1, :] += c[1]
        points = np.dot(R, points)

        # Return the points as a collection of tuples.
        capillary_points = []

        for k in range(0, points.shape[1]):
            capillary_points += [(points[0, k], points[1, k])]

        return capillary_points

    def get_fill_points(self, delta, sigma):

        if sigma < -1:
            raise ValueError('Sigma must be greater than -1.')

        # Point P: intersection of embedded capillary tube with cladding
        Py = (self.R_tube_center**2 + self.R_cladding**2 -
              (self.R_tube + self.T_tube)**2) / (2 * self.R_tube_center)
        Px = np.sqrt(self.R_cladding**2 - Py**2)

        # Point Q: location on exterior of capillary tube at which fill begins
        Qx = Px * np.cos(delta) + (Py - self.R_tube_center) * np.sin(delta)
        Qy = (Py - self.R_tube_center) * np.cos(delta) - \
            Px * np.sin(delta) + self.R_tube_center

        # Angle theta: angle away from vertical in which to go towards
        # cladding from point Q
        # This is chosen so that the angle with the capillary wall and
        # cladding wall are equal
        theta = np.arctan(Qx / (Qy - self.R_tube_center +
                          (self.R_tube_center *
                           (self.R_tube + self.T_tube) /
                           (self.R_cladding +
                            (self.R_tube + self.T_tube)))))

        # Unit vector e: points from Q toward cladding in direction we want
        ex, ey = np.sin(theta), np.cos(theta)

        # Dot products
        Qe = Qx * ex + Qy * ey
        QQ = Qx * Qx + Qy * Qy

        # Length t: distance from outer capillary wall to cladding along
        # (flat) fill
        t = -Qe + np.sqrt(Qe**2 + self.R_cladding**2 - QQ)

        # Point T: Point on inner cladding wall where fill begins
        Tx, Ty = Qx + t * ex, Qy + t * ey

        # Polar angle from y axis to point T on inner cladding
        nu = np.arccos(Ty / self.R_cladding)

        if nu > (np.pi/self.n_tubes):
            raise ValueError(
                "Value for fill angle delta too large, fills overlap.")

        # Vector TP
        TPx, TPy = Px - Tx, Py - Ty

        # Distance d: perpendicular distance from (flat) fill to line in
        # direction TP
        d = t/2 * np.sqrt((TPx * TPx + TPy * TPy)/(ex * TPx + ey * TPy)**2 - 1)

        # Control point c: lies along perpendicular to (flat) fill at
        # midpoint of fill.
        cx = Qx + t/2 * ex + d * sigma * ey
        cy = Qy + t/2 * ey - d * sigma * ex

        return (Tx, Ty), (cx, cy), (Qx, Qy), nu

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
