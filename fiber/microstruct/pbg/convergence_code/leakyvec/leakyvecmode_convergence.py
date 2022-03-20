#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General convergence study code.
"""
import numpy as np
import os
from fiberamp import FiberMode
from netgen.libngpy._meshing import NgException

z_exact = 3.7969194313378907-0.725805740812567j

# Folder setup.  Enter your path to pbg folder. ##################
folder = '/home/piet2/local/fiberamp/fiber/microstruct/pbg/outputs'

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("Given folder is not a directory. Make this \
directory and begin again.")

# Polynomial Degrees and Refinements to cycle through. #################

refs = [0, 1, 2, 3, 4, 5]
ps = [0, 1, 2, 3, 4, 5]
nspan = 6

Zs = np.zeros(shape=(len(refs), len(ps), nspan), dtype=complex)
dofs = np.zeros(shape=(len(refs), len(ps)), dtype=float)


if __name__ == '__main__':
    for i, ref in enumerate(refs):
        try:
            for j, p in enumerate(ps):
                print('\n' + '#'*8 + ' refinement: ' + str(ref) +
                      ', degree: ' + str(p) + '  ' + '#'*8 + '\n', flush=True)

                fbm = FiberMode(fibername='Nufern_Yb', R=3,
                                Rout=9, h=2, hcore=.3, refine=ref,
                                curveorder=max(p+1, 6))

                betas, zsqrs, E, phi, Robj = fbm.leakyvecmodes(ctr=z_exact,
                                                               rad=.005,
                                                               p=p,
                                                               alpha=1,
                                                               niterations=15,
                                                               nrestarts=0,
                                                               stop_tol=1e-9,
                                                               nspan=nspan)

                Zs[i, j, :len(zsqrs)] = zsqrs[:]
                dofs[i, j] = Robj.XY.ndof
        except NgException or Exception:
            print('\nMemory limit exceeded at ref: ', ref,
                  ', and p: ', p, '.\n Skipping rest of orders for this\
 refinement.', flush=True)
            pass
    print("\nLoops completed, saving data.\n", flush=True)

    np.save(os.path.abspath(folder + '/' + 'leakyvec_Zs'), Zs)
    np.save(os.path.abspath(folder + '/' + 'leakyvec_dofs'), dofs)
