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
    try:
        for i in range(ref):
            fiber_obj.mesh.Refine()
    except MemoryError:
        print("Unable to perform mesh refinements due to MemoryError.")
        return None, None, None

    try:
        z, y, _, beta, _, _ = fiber_obj.leakymode(p, rad=radius, ctr=center,
                                                  alpha=fiber_obj.alpha,
                                                  niterations=20, npts=npts,
                                                  nspan=nspan, nrestarts=1)
    except RuntimeError:
        print("Unable to complete modefinding due to RuntimeError.")
        return None, None, None

    return z, y, beta


if __name__ == '__main__':

    ps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    refs = [0, 1, 2]
    center = 1.242933-2.471929e-09j
    radius = .01

    folder = '/home/piet2/local/fiberamp/fiber/microstruct/pbg/\
outputs/lyr6cr2/convergence_studies'
    print(folder,'\n')
    if not os.path.isdir(os.path.relpath(folder)):
	    print('folder error.\n')
	    raise ValueError(
            "Given folder is not a directory.  Make this directory and begin\
 again.")

    else:
        for p in ps:
            for ref in refs:
                print('building A')
                A = PBG(params)
                z, _, beta = modefind(A, center, radius, p, ref)
                CL = 20 * beta.imag / np.log(10)

                if z is not None:
                    print("saving")
                    filename = 'p' + str(p) + '_refs' + str(ref)
                    filepath = os.path.abspath(folder + '/' + filename)

                    d = {'z': z, 'beta': beta,
                         'p': p, 'ref': ref, 'CL': CL}
                    print('saving\n')
                    np.savez(filepath, **d)
                    print('saving done, see if it worked.\n')
