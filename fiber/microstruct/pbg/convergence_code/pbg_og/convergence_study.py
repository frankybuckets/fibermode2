#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General convergence study code.

    To use:

    1: Enter desired fiber name and mode you wish to study. Make sure the
    correct center is listed.  Change import of parameters (below) to reflect
    fiber name.

    2: Enter your path to the pbg folder in fiberamp and create the output
    directory <pbg_home>/outputs/<fiber_name>/<mode_name>/convergence.  Or
    create directory of your choice and set it to variable <folder>.

    3: Alter polynomial degree list (ps) and refinement list (refs) to desired
    values. For our study, ps = [2,3,4,5,6,7,8,9,10,11] and refs = [0,1,2,3].
    This file is saved with ps = [1], refs = [0] for debugging purposes.
    Memory allocation errors are handled by passing to next iterate.

    4: Alter radius, nspan and npts as desired.  Center is set by mode name
    and shouldn't be changed.

After saving, alter script.h file appropriately and run sbatch script.h.
"""

from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params
import numpy as np
import os


# Fiber and mode names on which to perform convergence study. ##
fiber_name = 'lyr6cr2'   # Note: change import above to correspond.
mode_name = 'LP11'


# Starting location for eigenvalue iterations. #########################
centers = {'LP01': 1.242933 - 2.471929e-09j,
           'LP11': 1.93487063 - 8.699515e-08j}


# Folder setup.  Enter your path to pbg folder. ##################
pbg_home = '/home/pv/local/fiberamp/fiber/microstruct/pbg/'
folder = pbg_home + '/outputs/' + fiber_name + \
    '/' + mode_name + '/' + 'convergence'   # Make this directory

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("Given folder is not a directory. Make this \
directory and begin again.")

# Polynomial Degrees and Refinements to cycle through. #################
ps = [3]
refs = [0]


# Center, radius and span size for FEAST. ##############################
center = centers[mode_name]
radius = .001
nspan = 2                      # Number of initial eigenvectors
npts = 4                       # Number of quadrature points


if __name__ == '__main__':

    for p in ps:
        for ref in refs:
            print('Polynomial degree %i, with %i refinements: ' % (p, ref))
            print('Building fiber object and refining.\n')

            A = PBG(params)
            for i in range(ref):
                A.refine()

            print('Finding modes.\n')
            try:
                z, _, _, beta, P, _ = A.leakymode(p, rad=radius, ctr=center,
                                                  alpha=A.alpha,
                                                  niterations=20, npts=npts,
                                                  nspan=nspan, nrestarts=1)
                ndof = P.fes.ndof
                CL = 20 * beta.imag / np.log(10)
                d = {'z': z, 'beta': beta, 'p': p,
                     'ref': ref, 'CL': CL, 'ndofs': ndof}

                print('Saving data.\n')
                filename = 'p' + str(p) + '_refs' + str(ref)
                filepath = os.path.abspath(folder + '/' + filename)
                np.savez(filepath, **d)

            except MemoryError("Unable to find modes due to MemoryError."):
                pass
