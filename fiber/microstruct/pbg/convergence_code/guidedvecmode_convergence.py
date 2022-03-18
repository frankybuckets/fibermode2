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
import numpy as np
import os
from fiberamp import FiberMode

z_exact = -15.817570072953

# Folder setup.  Enter your path to pbg folder. ##################
pbg_home = '/home/piet2/local/fiberamp/fiber/microstruct/pbg/'
folder = pbg_home + '/outputs'   # Make this directory

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("Given folder is not a directory. Make this \
directory and begin again.")

# Polynomial Degrees and Refinements to cycle through. #################

refs = [0, 1]
ps = [0, 1]
nspan = 3

Zs = np.zeros(shape=(len(refs), len(ps), nspan), dtype=complex)
dofs = np.zeros(shape=(len(refs), len(ps)), dtype=float)


if __name__ == '__main__':
    for i, ref in enumerate(refs):
        for j, p in enumerate(ps):
            try:
                print('\n' + '#'*8 + ' refinement: ' + str(ref) +
                      ', degree: ' + str(p) + '  ' + '#'*8 + '\n')
                fbm = FiberMode(fibername='Nufern_Yb', R=2.5,
                                Rout=5, h=80, refine=ref,
                                curveorder=max(p+1, 4))

                betas, zsqrs, E, phi, Robj = fbm.guidedvecmodes(ctr=z_exact,
                                                                rad=.1,
                                                                p=p,
                                                                niterations=10,
                                                                nrestarts=0,
                                                                npts=6,
                                                                stop_tol=1e-9,
                                                                nspan=nspan)
                Zs[i, j, :len(zsqrs)] = zsqrs[:]
                dofs[i, j] = Robj.XY.ndof
            except MemoryError('\nMemory limit exceeded at ref: ', ref,
                               ', and p: ', p, '.\n Passing.'):
                pass

    print('Saving data.\n')

    np.save(os.path.abspath(folder + '/' + 'guidedvec_Zs'), Zs)
    np.save(os.path.abspath(folder + '/' + 'guidedvec_dofs'), dofs)
