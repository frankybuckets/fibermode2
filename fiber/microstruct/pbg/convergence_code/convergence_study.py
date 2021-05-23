#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:26:16 2021.

@author: pv

A general convergence study code for our fibers.

"""
from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params
import numpy as np
import os


def modefind(fiber_obj, center, radius, p, ref, nspan=2, npts=4):
    """Compute modes of fiber object."""
    for i in range(ref):
        fiber_obj.refine()

    z, y, _, beta, P, _ = fiber_obj.leakymode(p, rad=radius, ctr=center,
                                              alpha=fiber_obj.alpha,
                                              niterations=20, npts=npts,
                                              nspan=nspan, nrestarts=1)
    return z, y, beta, P.fes.ndof


if __name__ == '__main__':

    ps = [2, 3, 4, 5]         # Polynomial degrees to cycle through
    refs = [3]                # Refinements to cycle through

    center = 1.242933-2.471929e-09j    # Center for FEAST
    radius = .01                       # Radius for FEAST
    nspan = 2                          # Number of vectors for initial span

    folder = '/home/pv/local/fiberamp/fiber/microstruct/pbg/outputs/lyr6cr2/\
fund_mode/convergence'               # Folder for outputs

    if not os.path.isdir(os.path.relpath(folder)):

        raise ValueError(
            "Given folder is not a directory.  Make this directory and begin\
 again.")

    else:
        for p in ps:
            for ref in refs:
                print('Polynomial degree %i, with %i refinements: ' % (p, ref))
                print('Building fiber object.\n')
                A = PBG(params)

                print('Refining mesh and finding modes.\n')
                z, _, beta, ndof = modefind(A, center, radius, p, ref,
                                            nspan=nspan)

                print("Found modes.\n")
                CL = 20 * beta.imag / np.log(10)
                filename = 'p' + str(p) + '_refs' + str(ref)
                filepath = os.path.abspath(folder + '/' + filename)

                d = {'z': z, 'beta': beta,
                     'p': p, 'ref': ref, 'CL': CL, 'ndofs': ndof}
                print('Saving data.\n')
                np.savez(filepath, **d)
