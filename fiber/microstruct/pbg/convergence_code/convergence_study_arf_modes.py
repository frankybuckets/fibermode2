"""
convergence_study_arf_modes.py

Python file that runs p- and h-refinement convergence studies. Choices
include Poletti's ANF fiber and Kolyadin's 8-capillary fiber.
"""

import os
import sys
import pickle
import numpy as np
import prettytable
import ngsolve as ngs
import netgen
from fiberamp.fiber.microstruct import ARF, loadarf


@profile
def hausdorff_dist(U, V):
    """
    Function that computes the Hausdorff distance between two sets of
    complex numbers U and V (stored as 1D numpy arrays).
    """

    UmV = np.abs(U[:, np.newaxis] - V[np.newaxis, :])
    hUV = np.min(UmV, axis=1)
    hVU = np.min(UmV, axis=0)

    return max(np.max(hUV), np.max(hVU))


@profile
def convergence_study_arf_modes(totalews, ctr, rad, pmin, pmax, nref, nspan,
                                alpha=10, npts=4, eta_tol=1e-12,
                                stop_tol=1e-12, name='kolyadin', prefix=None):
    """
    convergence_study_arf_modes()

    INPUTS:

    totalews - The total number of eigenvalues (either a scalar
    integer or iterable of ints) for each center in ctrs. If FEAST
    picks up another eigenvalue outside of this, we either fill in
    the missing number of expected slots or grab the totalews
    eigenvalues closest to the center for FEAST.

    ctrs - The center of the circle of radius rad in the complex plane
    where the desired eigenvalue(s) is(/are) located.

    rad - The radius of the circular contour with center ctr.

    pmin - The minimum polynomial degree for the finite element space.

    pmax - The maximum polynomial degree for the finite element space.
    When pmax <= pmin, we perform a refinement study in h. Otherwise,
    we refine the mesh nref times and perform a refinement study in p.

    nref - The number of (uniform) mesh refinements to perform.

    nspan - The initial number of vectors to provide to FEAST.

    alpha - The PML strength (> 0) for the frequency-dependent
    eigenproblem.

    npts - The number of quadrature points (> 0) for FEAST.

    eta_tol - A parameter for FEAST, which we're leaving untouched for
    now.

    stop_tol - The (relative) stopping tolerance for FEAST.

    name - The name of the fiber, which right now is one of 'poletti'
    or 'kolyadin' (the default).

    prefix - A file prefix for preloading an ARF object. If None,
    a new ARF object will be created and saved.

    OUTPUTS:

    None
    """

    # Check if the outputs folder exists. If not, create it.
    if not os.path.isdir('outputs'):
        print('Making \'outputs\' subfolder in current directory.')
        os.mkdir('outputs')

    # If the prefix is None, make the prefix the name of a reference
    # ARF object.
    if prefix is None:
        a = ARF(name=name)
        print('Saving ARF object to file with prefix ' + name + '_reference')
        a.save(name + '_reference')
    else:
        # Attempt to load an ARF object from file.
        try:
            a = loadarf('outputs/' + prefix)
            print('Loaded arf from file outputs/' + prefix + '_arf.pkl')
        except Exception as e:
            print('Error loading ARF from file:', e)
            print('Creating and saving a new ARF object with ' +
                  'name=\'{0:s}\'.'.format(name))
            a = ARF(name=name)
            a.save(prefix)

    # Set the refinement study we are doing.
    study_type = 'h' if pmax <= pmin else 'p'

    # A list of polynomial degrees for the 'p' refinement study.
    ps = list(range(pmin, pmax + 1))

    # Lists for gathering information from the convergence studies.
    ndof = []
    ews = []
    bts = []
    hdiste = []
    hdistb = []
    closs = []

    # The center and radius, made digestible for the ARF.polyeig() method.
    ctrs = (ctr,)
    radi = (rad,)

    if study_type == 'h':
        # Set the polynomial degree for this convergence study.
        p = pmin

        for k in range(nref + 1):
            # Run the polynomial eigensolver for the ARF object.
            Zs, _, _, betas, P, _, _, _ = \
                a.polyeig(p=p, alpha=alpha, ctrs=ctrs, radi=radi,
                          nspan=nspan, npts=npts, eta_tol=eta_tol,
                          stop_tol=stop_tol)

            # Add the number of degrees of freedom.
            ndof.append(P.fes.ndof)

            # Grab the eigenvalues and propagation constants. First check to make sure
            # that we grabbed the correct number of eigenvalues. If not, adjust what
            # we computed to fit into what we asked for.
            if len(Zs[0]) < totalews:
                warn_str = 'Fewer than {0:d} eigenvalues, '.format(totalews) + \
                           'duplicating the final computed eigenvalue.'
                print(warn_str)

                # Since we came up short, fill out the remaining totalews - len(Zs[0])
                # slots with the last eigenvalue in Zs[0]. Do the same for the betas.
                temp = np.zeros(totalews, dtype=complex)
                temp[:len(Zs[0])] = Zs[0]
                temp[len(Zs[0]):] = Zs[0][-1]
                Zs = [temp.copy()]

                temp[:len(betas[0])] = betas[0]
                temp[len(betas[0]):] = betas[0][-1]
                betas = [temp.copy()]
            elif len(Zs) > totalews:
                isign = '-' if np.sign(ctrs[0].imag) < 0 else '+'
                warn_str = 'More than {:d} eigenvalue(s)'.format(totalews) + \
                           'found, grabbing {:d} '.format(totalews) + \
                           'nearest eigenvalue(s) to center ' + \
                           '{0} {1} {2}i.'.format(ctr.real, isign,
                                                  abs(ctr.imag))
                print(warn_str)

                # Grab the nearest eigenvalues to the center and the
                # corresponding propagation constants.
                sorted_indices = np.argsort(np.abs(Zs[0] - ctrs[0]))
                Zs = [Zs[0][sorted_indices[:totalews]]]
                betas = [betas[0][sorted_indices[:totalews]]]

            ews += Zs
            bts += betas

            # Compute the hausdorff distances between successive sets of eigenvalues
            # and propagation constants.
            if k > 0:
                hdiste.append(hausdorff_dist(ews[-1], ews[-2]))
                hdistb.append(hausdorff_dist(bts[-1], bts[-2]))

            # Compute the confinement loss.
            closs += [20 / np.log(10) * bts[-1].imag]

            # Uniformly refine the mesh.
            if k < nref:
                a.refine()
    else:
        for k in range(nref):
            a.refine()

        for p in ps:
            # Run the polynomial eigensolver for the ARF object.
            Zs, Ys, Yls, betas, P, longYs, longYls, _ = \
                a.polyeig(p=p, alpha=alpha, ctrs=ctrs, radi=radi,
                          nspan=nspan, npts=npts, eta_tol=eta_tol,
                          stop_tol=stop_tol)

            # Add the number of degrees of freedom.
            ndof.append(P.fes.ndof)

            # Grab the eigenvalues and propagation constants. First check to make sure
            # that we grabbed the correct number of eigenvalues. If not, adjust what
            # we computed to fit into what we asked for.
            if len(Zs[0]) < totalews:
                warn_str = 'Fewer than {0:d} eigenvalues, '.format(len(Zs)) + \
                           'duplicating the final computed eigenvalue.'
                print(warn_str)

                # Since we came up short, fill out the remaining totalews - len(Zs[0])
                # slots with the last eigenvalue in Zs[0].
                temp = np.zeros(totalews, dtype=complex)
                temp[:len(Zs[0])] = Zs[0]
                temp[len(Zs[0]):] = Zs[0][-1]
                Zs = [temp.copy()]

                temp[:len(betas[0])] = betas[0]
                temp[len(betas[0]):] = betas[0][-1]
                betas = [temp.copy()]
            elif len(Zs[0]) > totalews:
                isign = '-' if np.sign(ctrs[0].imag) < 0 else '+'
                warn_str = 'More than {:d} eigenvalue(s)'.format(totalews) + \
                           'found, grabbing {:d} '.format(totalews) + \
                           'nearest eigenvalue(s) to center ' + \
                           '{0} {1} {2}i.'.format(ctr.real, isign,
                                                  abs(ctr.imag))
                print(warn_str)

                # Grab the nearest eigenvalues to the center and the
                # corresponding propagation constants.
                sorted_indices = np.argsort(np.abs(Zs[0] - ctrs[0]))
                Zs = [Zs[0][sorted_indices[:totalews]]]
                betas = [betas[0][sorted_indices[:totalews]]]

            ews += Zs
            bts += betas

            # Compute the hausdorff distances between successive sets of eigenvalues
            # and propagation constants.
            if p > ps[0]:
                hdiste.append(hausdorff_dist(ews[-1], ews[-2]))
                hdistb.append(hausdorff_dist(bts[-1], bts[-2]))

            # Compute the confinement loss.
            closs += [20 / np.log(10) * bts[-1].imag]

    # Set the field names for the convergence study printout.
    field_names = [study_type, 'ndof']
    field_names += ['Z_' + study_type + str(k+1) for k in range(totalews)]
    field_names += ['H(Z, Z_{0})'.format(study_type)]
    field_names += ['beta_' + study_type + str(k+1) for k in range(totalews)]
    field_names += ['H(beta, beta_{0})'.format(study_type)]
    field_names += ['CL_{0}{1} (dB/m)'.format(study_type, k+1)
                    for k in range(totalews)]

    # Create a table with the computed (non-dimensional) eigenvalues,
    # propagation constantrs, and losses.
    table = prettytable.PrettyTable()
    table.field_names = field_names

    # An array of strings that indicate uniform refinement of the step sizes.
    hs = ['h_0']
    hs += ['h_0 / 2^{0}'.format(j) for j in range(1, nref + 1)]

    # An array of integers corresponding to the polynomial degrees.
    ps = list(range(pmin, pmax + 1))

    # Format strings for eigenvalues and propagation constants.
    zfmtstr = '{0:16.14f} {1} {2:16.14f}i'
    bfmtstr = '{0:18.8f} {1} {2:18.14f}i'

    def numfmt(fmtstr, x): return fmtstr.format(x.real,
                                                '-' if x.imag < 0 else '+',
                                                abs(x.imag))

    for k in range(len(ews)):
        row = []

        # Print the h- or p-refinement information based on the study.
        if study_type == 'h':
            row.append(hs[k])
        else:
            row.append(ps[k])

        # Add the number of degrees of freedom.
        row.append(ndof[k])

        # Add the computed non-dimensional eigenvalues.
        zstrs = [numfmt(zfmtstr, z) for z in ews[k]]
        row += zstrs

        # Add the (absolute) hausdorff distance between successive eigenvalue clusters.
        hdstr = '{0:.2e}'.format(hdiste[k - 1]) if k > 0 else '-'
        row.append(hdstr)

        # Add the computed propagation constants.
        bstrs = [numfmt(bfmtstr, b) for b in bts[k]]
        row += bstrs

        # Add the (absolute) hausdorff distance between propagation constants.
        hdstr = '{0:.2e}'.format(hdistb[k - 1]) if k > 0 else '-'
        row.append(hdstr)

        # Add the computed cladding loss.
        clstrs = ['{0:15.7e}'.format(cl) for cl in closs[k]]
        row += clstrs

        # Add the row to the table.
        table.add_row(row)

    # Title for the printout.
    diagnostic_title = '{0}-refinement study results of \'{1}\' fiber'

    print(diagnostic_title.format(study_type, name))
    print('-' * 79)
    print('\tp = {0},...,{1}'.format(pmin, pmax))
    print('\th = h_0,...,h_0/2^{0}'.format(nref))
    print('')
    print(table)


def init_params():
    """
    Method that sets up a parameter dictionary as inputs to the method
    convergence_study_arf_modes() method. Documentation of the below
    parameters can be found in the docstring of the method
    convergence_study_arf_modes().
    """

    # Default parameters for the ARF fiber.
    paramsdict = {
        'pmin': 2,
        'pmax': 4,
        'alpha': 10,
        'nref': 0,
        'nspan': 5,
        'npts': 4,
        'name': 'poletti',
        'prefix': None,
        'eta_tol': 1e-12,
        'stop_tol': 1e-12,
        'ctr': 0.0,
        'rad': 0.0,
        'totalews': 0
    }

    modestr = 'LP01'

    # A dictionary of centers and radii for the Poletti and Kolyadin
    # fibers. Each value in the poletti dictionary gives a center,
    # radius, and total (expected) number of eigenvalues to compute
    # for the given LP-like mode.
    ctr_rad_dict = {
        'poletti': {
            'LP01': (2.24, 0.01, 1),
            'LP11': (3.57, 0.01, 2),
            'LP21': (4.75, 0.01, 2),
            'LP02': (5.09, 0.01, 1)
        },
        'kolyadin': {
            'LP01': (2.35, 0.02, 1),
            'LP11': (3.68, 0.02, 2),  # these
            'LP21': (4.86, 0.02, 2),  # are
            'LP02': (5.20, 0.02, 1)   # untested
        }
    }

    # -------------------------------------------------------------------------
    # Command line arguments for the program. In order, they are:
    #
    #     Under construction.
    # -------------------------------------------------------------------------

    if len(sys.argv) == 2:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = paramdict['pmin']
    elif len(sys.argv) == 3:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
    elif len(sys.argv) == 4:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
    elif len(sys.argv) == 5:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
        paramsdict['alpha'] = float(sys.argv[4])
    elif len(sys.argv) == 6:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
        paramsdict['alpha'] = float(sys.argv[4])
        paramsdict['nspan'] = int(sys.argv[5])
    elif len(sys.argv) == 7:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
        paramsdict['alpha'] = float(sys.argv[4])
        paramsdict['nspan'] = int(sys.argv[5])
        paramsdict['npts'] = int(sys.argv[6])
    elif len(sys.argv) == 8:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
        paramsdict['alpha'] = float(sys.argv[4])
        paramsdict['nspan'] = int(sys.argv[5])
        paramsdict['npts'] = int(sys.argv[6])
        paramsdict['name'] = str(sys.argv[7])
    elif len(sys.argv) == 9:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
        paramsdict['alpha'] = float(sys.argv[4])
        paramsdict['nspan'] = int(sys.argv[5])
        paramsdict['npts'] = int(sys.argv[6])
        paramsdict['name'] = str(sys.argv[7])
        modestr = str(sys.argv[8])
    elif len(sys.argv) == 10:
        paramsdict['pmin'] = int(sys.argv[1])
        paramsdict['pmax'] = int(sys.argv[2])
        paramsdict['nref'] = int(sys.argv[3])
        paramsdict['alpha'] = float(sys.argv[4])
        paramsdict['nspan'] = int(sys.argv[5])
        paramsdict['npts'] = int(sys.argv[6])
        paramsdict['name'] = str(sys.argv[7])
        modestr = str(sys.argv[8])
        paramsdict['prefix'] = str(sys.argv[9])

    # Complain if the user attempts to run something unimplemented.
    name = paramsdict['name']

    if name == 'kolyadin' and modestr != 'LP01':
        err_str = 'Mode \'{:s}\' not implemented '.format(modestr) + \
                  'for \'{:s}\' fiber'.format(name)
        raise NotImplementedError(err_str)

    # Set the center, radius, and total expected eigenvalues for the center.
    ctr, rad, totalews = ctr_rad_dict[name][modestr]
    paramsdict['ctr'] = ctr
    paramsdict['rad'] = rad
    paramsdict['totalews'] = totalews

    return paramsdict, modestr


if __name__ == '__main__':
    # Grab the parameters from the command line.
    paramsdict, modestr = init_params()

    # Dump the parameter info to the user.
    print('Running convergence study for \'{0}\'-like '.format(modestr) +
          'mode(s) with the following parameters:')
    for k, v in paramsdict.items():
        print('{0:20s} : {1}'.format(k, v))
    print('-' * 79 + '\n')

    # Run the convergence study.
    convergence_study_arf_modes(**paramsdict)
