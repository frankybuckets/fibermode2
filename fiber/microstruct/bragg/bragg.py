#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:48:03 2022

@author: pv

"""
import netgen.geom2d as geom2d
from ngsolve import Mesh, CoefficientFunction
import numpy as np
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

        self.L = scale
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

    def create_mesh(self, ref=0, curve=8):
        """
        Create mesh from geometry.
        """
        self.mesh = Mesh(self.geo.GenerateMesh())

        self.refinements = 0
        for i in range(ref):
            self.mesh.ngmesh.Refine()

        self.refinements += ref
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh = Mesh(self.mesh.ngmesh.Copy())
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
        self.index = CoefficientFunction(self.refractive_indices)
        self.n0 = self.refractive_indices[-1]

        self.V = (self.L * self.k)**2 * (self.n0 ** 2 - self.index ** 2)
