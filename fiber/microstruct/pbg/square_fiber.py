#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues, June 7, 11:40:15 2022

@author: pv
"""
import ngsolve as ng
from numpy import pi
from fiberamp.fiber import ModeSolver
from netgen.geom2d import CSG2d, Circle, Rectangle, Solid2d
from netgen.geom2d import PointInfo as PI


class SquareFiber(ModeSolver):

    def __init__(self, Rsqr_in=.5, Rsqr_out=1.5, R=3, Rout=4.5,
                 refine=0, curve=3, corner_maxh=.1):

        self.Rsqr_in = Rsqr_in
        self.Rsqr_out = Rsqr_out

        self.R = R
        self.Rout = Rout
        self.corner_maxh = corner_maxh

        geo = CSG2d()
        core = Solid2d([(-Rsqr_in, -Rsqr_in), PI(maxh=corner_maxh),
                        (Rsqr_in, -Rsqr_in), PI(maxh=corner_maxh),
                        (Rsqr_in, Rsqr_in), PI(maxh=corner_maxh),
                        (-Rsqr_in, Rsqr_in), PI(maxh=corner_maxh)],
                       mat="core")

        square_clad = Rectangle(pmin=(-Rsqr_out, -Rsqr_out),
                                pmax=(Rsqr_out, Rsqr_out),
                                mat="square_clad")

        outer_clad = Circle(center=(0, 0), radius=self.R, mat='outer_clad')
        Outer = Circle(center=(0, 0), radius=self.Rout, mat='Outer',
                       bc='OuterCircle')

        core.Maxh(.2)
        square_clad.Maxh(.5)
        outer_clad.Maxh(1)
        Outer.Maxh(2)

        geo.Add(core)
        geo.Add(square_clad - core)
        geo.Add(outer_clad - square_clad)
        geo.Add(Outer - outer_clad)

        self.geo = geo
        self.spline_geo = geo.GenerateSplineGeometry()
        self.mesh = ng.Mesh(geo.GenerateMesh())
        for r in range(refine):
            self.mesh.ngmesh.Refine()
        self.mesh.ngmesh.SetGeometry(self.spline_geo)
        self.mesh.Curve(curve)

        self.n0 = 1.45
        self.n_core = 1.48
        self.n_clad = 1.45
        self.wavelength = .5e-6
        self.k = 2*pi/self.wavelength
        self.L = 1.2e-6

        self.N = ng.CoefficientFunction([self.n_core, self.n_clad,
                                         self.n_clad, self.n0])
        self.V = (self.k * self.L)**2 * (self.n0**2 - self.N**2)
        self.index = self.N

        super().__init__(self.mesh, self.L, self.n0)
