#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 19:35:05 2021

@author: pv
"""
import numpy as np
import matplotlib.pyplot as plt
from ngsolve.special_functions import jv as ng_jv
from ngsolve.special_functions import hankel1 as ng_hankel1
from ngsolve.special_functions import hankel2 as ng_hankel2
from ngsolve.special_functions import kv as ng_kv

from ngsolve import x, y, log, sqrt

r = sqrt(x * x + y * y)
theta = log(x + 1j * y).imag

# Extend ngsolve functions to negative orders


def ng_yv(z, nu):
    if nu >= 0:
        return 1/(2j)*(ng_hankel1(z, nu) - ng_hankel2(z, nu))
    else:
        if nu == np.floor(nu):
            return 1/(2j)*(ng_hankel1(z, nu) - ng_hankel2(z, nu)) * \
                np.cos(np.pi * nu)
        else:
            return 1/(2j)*(ng_hankel1(z, nu) - ng_hankel2(z, nu)) * \
                np.cos(np.pi * nu) + ng_jv(z, nu) * np.sin(np.pi * nu)


def Jv(z, nu):
    if nu >= 0:
        return ng_jv(z, nu)
    else:
        sf = (nu/abs(nu)) ** int(abs(nu))
        return sf * ng_jv(z, int(abs(nu)))


def Yv(z, nu):
    if nu >= 0:
        return ng_yv(z, nu)
    else:
        sf = (nu/abs(nu)) ** int(abs(nu))
        return sf * ng_yv(z, int(abs(nu)))


def Kv(z, nu):
    if nu >= 0:
        return ng_kv(z, nu)
    else:
        sf = (nu/abs(nu)) ** int(abs(nu))
        return sf * ng_kv(z, int(abs(nu)))


def Hankel1(z, nu):
    if nu >= 0:
        return ng_hankel1(z, nu)
    else:
        sf = (nu/abs(nu)) ** int(abs(nu))
        return sf * ng_hankel1(z, int(abs(nu)))


def Hankel2(z, nu):
    if nu >= 0:
        return ng_hankel2(z, nu)
    else:
        sf = (nu/abs(nu)) ** int(abs(nu))
        return sf * ng_hankel2(z, int(abs(nu)))

# Derivative of jv and hankel1 (as ngsolve coefficient functions)


def Jvp(z, nu):
    return .5 * (Jv(z, nu - 1) - Jv(z, nu + 1))


def Yvp(z, nu):
    return .5 * (Yv(z, nu - 1) - Yv(z, nu + 1))


def Jvpp(z, nu):
    return .25 * (Jv(z, nu - 2) - 2 * Jv(z, nu) + Jv(z, nu + 2))


def Yvpp(z, nu):
    return .25 * (Yv(z, nu - 2) - 2 * Yv(z, nu) + Yv(z, nu + 2))


def Hankel1p(z, nu):
    return .5 * (Hankel1(z, nu - 1) - Hankel1(z, nu + 1))


def Hankel1pp(z, nu):
    return .25 * (Hankel1(z, nu - 2) - 2 * Hankel1(z, nu) + Hankel1(z, nu + 2))


def Hankel2p(z, nu):
    return .5 * (Hankel2(z, nu - 1) - Hankel2(z, nu + 1))


def Hankel2pp(z, nu):
    return .25 * (Hankel2(z, nu - 2) - 2 * Hankel2(z, nu) + Hankel2(z, nu + 2))

# Plotting utility


def plotlogf(f, rmin, rmax, imin, imax, *args, title='',
             levels=25, truncate=False, h=2, equal=False,
             colorbar=True, rref=20, iref=20, figsize=(14, 7),
             log_off=False, loop=False, three_D=False,
             phase=False, R2plane=False):
    """Create contour plot of complex function f on given range."""
    Xr = np.linspace(rmin, rmax, num=rref)
    Xi = np.linspace(imin, imax, num=iref)
    xr, xi = np.meshgrid(Xr, Xi)
    if not R2plane:
        zs = xr + 1j * xi
    if not loop:
        if not R2plane:
            fx = f(zs, *args)
        else:
            fx = f(xr, xi, *args)

    else:
        if not R2plane:
            fx = np.zeros_like(zs)
            for i in range(zs.shape[0]):
                for j in range(zs.shape[1]):
                    fx[i, j] = f(zs[i, j], *args)
        else:
            fx = np.zeros_like(xr)
            for i in range(zs.shape[0]):
                for j in range(zs.shape[1]):
                    fx[i, j] = f(xr[i, j], xi[i, j], *args)
        # i, j = 0, 0
        # while i < zs.shape[0]:
        #     fx[i, j] = f(zs[i, j], *args)
        #     j += 1
        #     if (j == zs.shape[1]):
        #         j = 0
        #         i += 1
    if truncate:
        fx[np.abs(fx) > h] = h  # ignore large values to focus on roots

    if phase:
        if R2plane:
            raise ValueError('Working with R2plane=True; output assumed real, \
hence has no phase.  If complex output desired set R2plane=False (default).')
        ys = np.angle(fx)
    else:
        if R2plane:
            ys = fx
        else:
            ys = np.abs(fx)

    if not log_off and phase:
        raise ValueError("Need log_off when phase is turned on.")

    if not log_off:
        ys = np.log(np.abs(fx))

    fig = plt.figure(figsize=figsize)

    if three_D:
        ax = fig.add_subplot(projection='3d')
        lims = (np.ptp(Xr), np.ptp(Xi), min(np.ptp(Xr), np.ptp(Xi)))
        ax.set_box_aspect(lims)
        ax.set_axis_off()
        # ax.set_xlim3d(auto=True)
        # ax.set_ylim3d(auto=True)

        # ax.autoscale_view(tight=True)

    else:
        ax = fig.add_subplot()
        ax.grid(True)
        ax.set_facecolor('grey')

    if equal:
        ax.axis('equal')
        # plt.figure(figsize=(1.2 * (rmax - rmin), imax - imin))

    im = ax.contour(xr, xi, ys, levels=levels)

    plt.title(title)
    if colorbar:
        plt.colorbar(im)


def plotlogf_real(f, x_min, x_max, *args, n=1000, figsize=(12, 8),
                  level=np.e, log_off=False, truncate=False, height=2,
                  bounds=None):
    """Create contour plot of complex function f on given range."""

    xs = np.linspace(x_min, x_max, num=n)
    fx = f(xs, *args)

    plt.figure(figsize=figsize)

    if log_off:
        if truncate:
            fx[np.abs(fx) > height] = height  # truncate to height
        ys = np.abs(fx)
        plt.plot(xs, ys)
    else:
        ys = np.zeros_like(fx, dtype=float)
        ys[np.abs(fx) >= level] = np.log(np.abs(fx[np.abs(fx) >= level]))
        ys[np.abs(fx) < level] = np.abs(
            fx[np.abs(fx) < level]) * (np.log(level) / level)
        if truncate:
            ys[np.abs(ys) > height] = height
        plt.plot(xs, ys)

    bottom, top = plt.ylim()
    plt.ylim(0, top)

    if bounds is not None:
        l, r = bounds[:]
        plt.plot([l, l], [0, top], color='orange', linestye=':', linewidth=.75)
        plt.plot([r, r], [0, top], color='orange', linestye=':', linewidth=.75)
    plt.grid(True)
