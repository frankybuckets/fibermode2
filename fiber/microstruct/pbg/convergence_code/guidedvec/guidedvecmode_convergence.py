#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General convergence study code.
"""

import numpy as np
import os
from fiberamp import FiberMode
from netgen.libngpy._meshing import NgException

z_exact = -15.817570072953

# Folder setup.  Enter your path to pbg folder. ##################
folder = '/home/piet2/local/fiberamp/fiber/microstruct/pbg/outputs'

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("Given folder is not a directory. Make this \
directory and begin again.")

# Polynomial Degrees and Refinements to cycle through. #################

refs = [0, 1, 2, 3, 4]
ps = [0, 1, 2, 3, 4]
nspan = 3

Zs = np.zeros(shape=(len(refs), len(ps), nspan), dtype=complex)
dofs = np.zeros(shape=(len(refs), len(ps)), dtype=float)


if __name__ == '__main__':
    for i, ref in enumerate(refs):
        try:
            for j, p in enumerate(ps):
                print('\n' + '#'*8 + ' refinement: ' + str(ref) +
                      ', degree: ' + str(p) + '  ' + '#'*8 + '\n', flush=True)
                fbm = FiberMode(fibername='Nufern_Yb', R=2.5,
                                Rout=5, h=80, refine=ref,
                                curveorder=max(p+1, 6))

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
                print("guided vec modes complete, saving.\n", flush=True)
                np.save(os.path.relpath(folder + '/' + 'guidedvec_Zs'), Zs)
                np.save(os.path.relpath(folder + '/' + 'guidedvec_dofs'), dofs)
        except (NgException, Exception):
            print('\nMemory limit exceeded at ref: ', ref,
                  ', and p: ', p, '.\n Skipping rest of orders for this\
 refinement.')
            pass

    print('Saving data again just in case.\n', flush=True)

    np.save(os.path.relpath(folder + '/' + 'guidedvec_Zs'), Zs)
    np.save(os.path.relpath(folder + '/' + 'guidedvec_dofs'), dofs)
