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


# Polynomial Degrees and Refinements to cycle through. #################
ps = [2, 3, 4]
refs = [0, 1]


if __name__ == '__main__':

    for p in ps:
        for ref in refs:
            print(p, ref)
