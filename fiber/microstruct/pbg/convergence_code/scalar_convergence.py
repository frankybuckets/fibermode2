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
import ngsolve as ng
from netgen.geom2d import SplineGeometry as SG
from pyeigfeast import SpectralProjNG, NGvecs
import numpy as np
import os

R = 2.5

# Folder setup.  Enter your path to pbg folder. ##################
pbg_home = '/home/pv/local/fiberamp/fiber/microstruct/pbg/'
folder = pbg_home + '/outputs'   # Make this directory

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("Given folder is not a directory. Make this \
directory and begin again.")

# Polynomial Degrees and Refinements to cycle through. #################
refs = [0, 1, 2, 3, 4, 5, 6, 7]
ps = [1, 2, 3, 4, 5]

Zs = np.zeros(shape=(len(refs), len(ps)))
dofs = np.zeros_like(Zs)


if __name__ == '__main__':

    for i, ref in enumerate(refs):

        # Reset to coarsest mesh
        geo = SG()
        geo.AddCircle(c=(0, 0), r=R, bc='outer')
        mesh = ng.Mesh(geo.GenerateMesh(maxh=1))
        mesh.Curve(3)

        for r in range(ref):  # refine 'ref' number of times
            print('\n #### refining!! #### \n')
            mesh.Curve(0)
            mesh.ngmesh.Refine()
            mesh.Curve(3)

        for j, p in enumerate(ps):
            try:
                print('\n' + '#'*8 + ' refinement: ' + str(ref) +
                      ', degree: ' + str(p) + '  ' + '#'*8 + '\n')

                fes = ng.H1(mesh, order=p, complex=True, dirichlet='outer')
                dofs[i, j] = fes.ndof

                u, v = fes.TnT()

                A = ng.BilinearForm(fes)
                A += -ng.grad(u) * ng.grad(v) * ng.dx

                B = ng.BilinearForm(fes)
                B += u * v * ng.dx

                with ng.TaskManager():
                    A.Assemble()
                    B.Assemble()

                P = SpectralProjNG(fes, A.mat, B.mat, center=-
                                   0.92530975, radius=.1, npts=6)

                nspan = 2

                Y = NGvecs(fes, nspan)
                Y.setrandom(seed=1)

                zsqrs, Y, hist, _ = P.feast(
                    Y, hermitian=True, nrestarts=0, niterations=30,
                    stop_tol=1e-12)

                Zs[i, j] = zsqrs[0]

            except MemoryError('\nMemory limit exceeded at ref: ', ref,
                               ', and p: ', p, '.\n'):
                pass

    print('Saving data.\n')

    np.save(os.path.abspath(folder + '/' + 'scalar_Zs'), Zs)
    np.save(os.path.abspath(folder + '/' + 'dofs'), dofs)
