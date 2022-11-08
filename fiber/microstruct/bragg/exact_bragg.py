#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:48:03 2022

@author: pv

"""
import numpy as np
import netgen.geom2d as geom2d
import ngsolve as ng

from ngsolve import x, y, exp, CF
from opticalmaterialspy import Air, SiO2

from scipy.special import jv, jvp, h1vp, h2vp, yv, yvp
from scipy.special import hankel1 as h1
from scipy.special import hankel2 as h2

from step_exact.step_exact_utility_functions import r, theta, \
    Jv, Yv, Hankel1, Hankel2, Jvp, Yvp, Hankel1p, Hankel2p


class BraggExact():
    """
    Create a Bragg type multilayered radial fiber.

    Bragg fibers consist of a circular core region surrounded by many
    concentric circular layers of alternating material, often glass and air.

    """

    def __init__(self, scale=15e-6, ts=[15e-6, 15*.5e-6, 15e-6, 15*.5e-6],
                 mats=['air', 'glass', 'air', 'Outer'],
                 ns=[Air().n, SiO2().n, Air().n, SiO2().n],
                 maxhs=[.4, .1, .1, .1],
                 wl=1.8e-6, ref=0, curve=8):

        self.scale = scale
        self.ts = np.array(ts)
        self.mats = mats
        self.rhos = np.array([sum(ts[:i]) for i in range(1, len(ts)+1)])
        self.maxhs = np.array(maxhs) * self.rhos / scale

        self.wavelength = wl
        self.k0 = 2 * np.pi / self.wavelength
        self.ns = np.array([ns[i](wl) for i in range(len(ns))])
        self.ks = self.k0 * self.ns

        # Create geometry
        self.create_geometry()
        self.create_mesh(ref=ref, curve=curve)

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

        Rs = self.rhos / self.scale
        for i, R in enumerate(Rs[:-1]):
            self.geo.AddCircle(c=(0, 0), r=R, leftdomain=i+1,
                               rightdomain=i+2)

        self.geo.AddCircle(c=(0, 0), r=Rs[-1], leftdomain=len(Rs),
                           bc='OuterCircle')

        for i, (mat, maxh) in enumerate(zip(self.mats, self.maxhs)):
            self.geo.SetMaterial(i+1, mat)
            self.geo.SetDomainMaxH(i+1, maxh)

    def transfer_matrix(self, beta, nu, rho, n1, n2, zfunc='bessel'):
        """Return transfer matrix from Yeh et al Theory of Bragg Fiber.

        We have scaled the H field such that we make the replacements:

            we ---> k0n^2,  wu ---> k0,  e1/e2 ---> (n1/n2)^2

        We take all u to be idential to u0. This function also assumes
        that provided beta is scaled, but the provided rho is not scaled."""

        beta = np.array(beta, dtype=np.complex128)

        M = np.zeros(beta.shape + (4, 4), dtype=np.complex128)

        k0 = self.k0 * self.scale
        k1 = k0 * n1
        k2 = k0 * n2

        rho = rho / self.scale

        K1 = np.sqrt(k1 ** 2 - beta ** 2, dtype=complex)
        K2 = np.sqrt(k2 ** 2 - beta ** 2, dtype=complex)

        X, Y = K1*rho,  K2*rho

        expr = (1 / Y - Y / X**2)

        F1 = (K2 * n1 ** 2) / (K1 * n2 ** 2)
        F2 = nu * beta / (k0 * n2 ** 2)
        F3 = nu * beta / k0
        F4 = K2 / K1

        Ymat = np.zeros_like(M)
        Ymat[..., 0, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T
        Ymat[..., 1, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T
        Ymat[..., 2, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T
        Ymat[..., 3, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T

        if zfunc == 'hankel':
            z1, z1p = h1, h1vp
            z2, z2p = h2, h2vp
        elif zfunc == 'bessel':
            z1, z1p = jv, jvp
            z2, z2p = yv, yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        M[..., 0, :] = np.array([(z1(nu, X) * z2p(nu, Y) -
                                  F1 * z1p(nu, X) * z2(nu, Y)).T,

                                 (z2(nu, X) * z2p(nu, Y) -
                                 F1 * z2p(nu, X) * z2(nu, Y)).T,

                                 (1j*F2 * z1(nu, X) * z2(nu, Y)*expr).T,

                                 (1j*F2 * z2(nu, X) * z2(nu, Y)*expr).T]).T

        M[..., 1, :] = np.array([(F1 * z1p(nu, X) * z1(nu, Y) -
                                z1(nu, X) * z1p(nu, Y)).T,

                                 (F1 * z2p(nu, X) * z1(nu, Y) -
                                z2(nu, X) * z1p(nu, Y)).T,

                                 (-1j*F2 * z1(nu, X) * z1(nu, Y)*expr).T,

                                 (-1j*F2 * z2(nu, X) * z1(nu, Y)*expr).T]).T

        M[..., 2, :] = np.array([(-1j * F3 * z1(nu, X) * z2(nu, Y) * expr).T,

                                 (-1j * F3 * z2(nu, X) * z2(nu, Y) * expr).T,

                                 (z1(nu, X) * z2p(nu, Y) - F4 *
                                 z1p(nu, X) * z2(nu, Y)).T,

                                 (z2(nu, X) * z2p(nu, Y) - F4 *
                                  z2p(nu, X) * z2(nu, Y)).T]).T

        M[..., 3, :] = np.array([(1j * F3 * z1(nu, X) * z1(nu, Y) * expr).T,

                                 (1j * F3 * z2(nu, X) * z1(nu, Y) * expr).T,

                                 (F4 * z1p(nu, X) * z1(nu, Y) -
                                 z1(nu, X) * z1p(nu, Y)).T,

                                 (F4 * z2p(nu, X) * z1(nu, Y) -
                                 z2(nu, X) * z1p(nu, Y)).T]).T

        return np.pi / 2 * Ymat * M

    def state_matrix(self, beta, nu, rho, n, zfunc='bessel'):
        """Return matching matrix from Yeh et al Theory of Bragg Fiber.

        This matrix appears as equation 34 in that paper. Again we have scaled
        the H field such that we make the replacements:

            we ---> k0n^2,  wu ---> k0,  e1/e2 ---> (n1/n2)^2

        We take all u to be idential to u0. This function also assumes
        that provided beta is scaled, but the provided rho is not scaled."""

        beta = np.array(beta, dtype=np.complex128)

        k0 = self.k0 * self.scale
        k = k0 * n
        K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)

        rho /= self.scale

        L = np.zeros(beta.shape + (4, 4), dtype=complex)
        Z = np.zeros_like(beta).T

        if zfunc == 'hankel':
            z1, z1p = h1, h1vp
            z2, z2p = h2, h2vp
        elif zfunc == 'bessel':
            z1, z1p = jv, jvp
            z2, z2p = yv, yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        L[..., 0, :] = np.array([z1(nu, K * rho).T, z2(nu, K * rho).T,
                                 Z, Z]).T

        L[..., 1, :] = np.array([(k0 * n**2 / (beta * K) * z1p(nu, K * rho)).T,
                                 (k0 * n**2 / (beta * K) * z2p(nu, K * rho)).T,
                                 (1j*nu / (K**2 * rho) * z1(nu, K * rho)).T,
                                 (1j*nu / (K**2 * rho) * z2(nu, K * rho)).T]).T

        L[..., 2, :] = np.array([Z, Z, z1(nu, K * rho).T, z2(nu, K * rho).T]).T

        L[..., 3, :] = np.array([(1j*nu / (K**2 * rho) * z1(nu, K * rho)).T,
                                 (1j*nu / (K**2 * rho) * z2(nu, K * rho)).T,
                                 (-k0 / (beta * K) * z1p(nu, K * rho)).T,
                                 (-k0 / (beta * K) * z2p(nu, K * rho)).T]).T

        return L

    def state_matrix_inverse(self, beta, nu, rho, n, zfunc='bessel'):
        """Return inverse of matching matrix above.

        Same replacements as above.  Inverse found using sympy.  Also uses
        K * rho = x.  Mainly for debugging transfer matrix."""

        beta = np.array(beta, dtype=np.complex128)

        k0 = self.k0 * self.scale
        k = k0 * n
        K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)

        rho /= self.scale

        X = K * rho
        F1 = K * beta / (k0 * n**2)
        F2 = beta * nu / (k0 * n**2 * X)
        F3 = beta * nu / (k0 * X)
        F4 = K * beta / k0

        L = np.zeros(beta.shape + (4, 4), dtype=complex)
        Z = np.zeros_like(beta).T

        if zfunc == 'hankel':
            z1, z1p = h1, h1vp
            z2, z2p = h2, h2vp
        elif zfunc == 'bessel':
            z1, z1p = jv, jvp
            z2, z2p = yv, yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        L[..., 0, :] = np.array([z2p(nu, X).T,
                                 (-F1 * z2(nu, X)).T,
                                 (1j * F2 * z2(nu, X)).T,
                                 Z]).T

        L[..., 1, :] = np.array([(-z1p(nu, X)).T,
                                 (F1 * z1(nu, X)).T,
                                 (-1j * F2 * z1(nu, X)).T,
                                 Z]).T

        L[..., 2, :] = np.array([(-1j * F3 * z2(nu, X)).T,
                                 Z,
                                 z2p(nu, X).T,
                                 (F4 * z2(nu, X)).T]).T

        L[..., 3, :] = np.array([(-1j * F3 * z1(nu, X)).T,
                                 Z,
                                 -z1p(nu, X).T,
                                 (-F4 * z1(nu, X)).T]).T

        L[..., :, :] = np.pi * X / 2 * L[..., :, :]

        return L

    def determinant(self, beta, nu=1, outer='h2', return_coeffs=False,
                    return_matrix=False):
        """Return determinant of matching matrix.

        Provided beta should be scaled.  This zeros of this functions are the
        propagation constants for the fiber."""

        if return_coeffs and return_matrix:
            raise ValueError("Only one of return_matrix and return_coeffs\
 can be set to True")

        beta = np.array(beta, dtype=np.complex128)

        rhos = self.rhos
        ns = self.ns
        L = np.zeros(beta.shape + (4, 2), dtype=complex)
        L[..., :, :] = np.eye(4)[:, [0, 2]]  # pick J columns for core

        for i in range(len(rhos)-2):
            nl, nr = ns[i], ns[i+1]
            rho = rhos[i]
            L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L

        L = self.state_matrix(beta, nu, rhos[-2], ns[-2]) @ L

        if outer == 'h2':
            inds = [1, 3]
        elif outer == 'h1':
            inds = [0, 2]
        else:
            raise TypeError("Outer function must be 'h1' (guided) or 'h2'\
 (leaky).")

        R = self.state_matrix(beta, nu, rhos[-2],
                              ns[-1], zfunc='hankel')[..., inds]

        a, b, e, f = L[..., 0, 0], L[..., 0, 1], L[..., 1, 0], L[..., 1, 1]
        c, d, g, h = L[..., 2, 0], L[..., 2, 1], L[..., 3, 0], L[..., 3, 1]

        alpha, Beta = R[..., 0, 0], R[..., 2, 1]
        gamma, delta = R[..., 1, 0], R[..., 1, 1]
        epsilon, sigma = R[..., 3, 0], R[..., 3, 1]

        A = e - (a/alpha * gamma + c/Beta * delta)
        B = f - (b/alpha * gamma + d/Beta * delta)
        C = g - (a/alpha * epsilon + c/Beta * sigma)
        D = h - (b/alpha * epsilon + d/Beta * sigma)

        if return_coeffs:
            return A, B, C, D, a, b, c, d, alpha, Beta
        else:
            if return_matrix:
                M = np.zeros(beta.shape + (4, 4), dtype=complex)
                M[..., : 2] = L
                M[..., 2:] = R
                return M
            else:
                return C * B - A * D

    def coefficients(self, beta, nu=1, outer='h2'):
        """Return coefficients for fields of bragg fiber.

        Returns an array where each row gives the coeffiencts of the fields
        in the associated region of the fiber."""

        A, B, C, D, a, b, c, d, \
            alpha, Beta = self.determinant(beta, nu, outer,
                                           return_coeffs=True)

        # A, B, or C,D can be used to make v1,v2 for core, but if mode
        # is transverse one of thes pairs is zero, below we account for this
        Vs = np.array([A, B, C, D])
        Vn = ((Vs*Vs.conj()).real) ** .5

        if max(Vn) < 1e-13:  # Check that not all are too small
            raise ValueError("Error in coefficients: small matrix elements\
 are all too small.")

        imax = np.argmax(Vn)

        if imax in [0, 1]:
            print("Using A,B")
            v1, v2 = B, -A  # If C,D too small, assign with B,A

        else:
            print("using C, D")
            v1, v2 = D, -C  # Otherwise use C and D

        v = np.array([v1, v2])

        w1, w2 = (a*v1 + b*v2) / alpha, (c*v1 + d*v2) / Beta

        rhos = self.rhos
        ns = self.ns

        M = np.zeros((len(self.rhos), 4), dtype=complex)
        L = np.zeros((4, 2), dtype=complex)

        L[:, :] = np.eye(4)[:, [0, 2]]
        L = L @ v
        M[0, :] = L

        for i in range(len(rhos)-2):
            nl, nr = ns[i], ns[i+1]
            rho = rhos[i]
            L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L
            M[i+1, :] = L

        if outer == 'h2':
            inds = [1, 3]
        elif outer == 'h1':
            inds = [0, 2]
        else:
            raise TypeError("Outer function must be 'h1' (guided) or 'h2'\
     (leaky).")

        M[-1, inds] = w1, w2

        if nu > 0:  # Hybrid, Ez non zero in core
            return 1 / v1 * M  # Normalize so Ez coeff in core is 1

        else:  # TE or TM,
            # Find non-zero coefficient (either for Ez or Hz) and use to scale
            vscale = v[np.argmax((v*v.conj()).real)]
            return 1/vscale * M

    def regional_fields(self, beta, coeffs, index, nu=1, outer='h2',
                        zfunc='bessel'):
        """Create fields on one region of the fiber."""

        A, B, C, D = coeffs[:]

        k0 = self.k0 * self.scale
        n = self.ns[index]
        k = k0 * n

        K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)
        F = 1j * beta / (k ** 2 - beta ** 2)

        if zfunc == 'hankel':
            Z1, Z1p = Hankel1, Hankel1p
            Z2, Z2p = Hankel2, Hankel2p
        elif zfunc == 'bessel':
            Z1, Z1p = Jv, Jvp
            Z2, Z2p = Yv, Yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        einu = exp(1j * nu * theta)

        Ez = (A * Z1(K*r, nu) + B * Z2(K*r, nu)) * einu
        dEzdr = K * (A * Z1p(K*r, nu) + B * Z2p(K*r, nu)) * einu
        dEzdt = 1j * nu * Ez

        Hz = (C * Z1(K*r, nu) + D * Z2(K*r, nu)) * einu
        dHzdr = K * (C * Z1p(K*r, nu) + D * Z2p(K*r, nu)) * einu
        dHzdt = 1j * nu * Hz

        Er = F * (dEzdr + k0 / (beta * r) * dHzdt)
        Ephi = F * (1 / r * dEzdt - k0 / beta * dHzdr)

        Hr = F * (dHzdr - k0 * n**2 / (beta * r) * dEzdt)
        Hphi = F * (1 / r * dHzdt + k0 * n**2 / beta * dEzdr)

        Ex = (x * Er - y * Ephi) / r
        Ey = (y * Er + x * Ephi) / r

        Hx = (x * Hr - y * Hphi) / r
        Hy = (y * Hr + x * Hphi) / r

        return {'Ez': Ez, 'Hz': Hz, 'Er': Er, 'Hr': Hr, 'Ephi': Ephi,
                'Hphi': Hphi,  'Ex': Ex, 'Ey': Ey, 'Hx': Hx, 'Hy': Hy}

    def all_fields(self, beta, nu=1, outer='h2'):
        """Create total fields for fiber from regional fields."""
        M = self.coefficients(beta, nu, outer)

        Ez, Hz = [], []
        Er, Hr = [], []
        Ephi, Hphi = [], []
        Ex, Ey, Hx, Hy = [], [], [], []

        for i in range(len(self.ns)):
            coeffs = M[i]
            if i == len(self.ns) - 1:
                zfunc = 'hankel'
            else:
                zfunc = 'bessel'
            F = self.regional_fields(beta, coeffs, i, nu, outer, zfunc=zfunc)
            Ez.append(F['Ez']), Er.append(F['Er'])
            Ex.append(F['Ex']), Ey.append(F['Ey'])
            Ephi.append(F['Ephi'])
            Hz.append(F['Hz']), Hr.append(F['Hr'])
            Hx.append(F['Hx']), Hy.append(F['Hy'])
            Hphi.append(F['Hphi'])

        Ez, Er, Ephi, Ex, Ey = CF(Ez), CF(Er), CF(Ephi), CF(Ex), CF(Ey)
        Hz, Hr, Hphi, Hx, Hy = CF(Hz), CF(Hr), CF(Hphi), CF(Hx), CF(Hy)

        # Transverse Fields

        Etv = CF((Ex, Ey))
        Htv = CF((Hx, Hy))

        return {'Ez': Ez, 'Hz': Hz, 'Er': Er, 'Hr': Hr,
                'Ephi': Ephi, 'Hphi': Hphi, 'Etv': Etv, 'Htv': Htv,
                'Ex': Ex, 'Ey': Ey, 'Hx': Hx, 'Hy': Hy,
                }
