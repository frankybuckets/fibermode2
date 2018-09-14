"""
This file collects the often used data members of step-index optical
fibers into a class. Methods are provided to calculate propagation
constants and fiber modes.
"""

import scipy.special as scf
from math import pi, sqrt, atan2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.optimize import fsolve, bisect


class Fiber:

    """ A class of step-index cylindrical fibers """

    def __init__(self, case=None,
                 L=None, rcore=None, rclad=None, nclad=None, ncore=None,
                 ks=None):
        """
        A step-index fiber object F can be made using a preprogrammed case
        name (such as 'artifical_singlemode', 'Nufern', etc) by
           F = Fiber(case).
        Alternately, a fiber object can be made by providing these kwargs:
           L:      total fiber length
           rcore:  core cross-section radius
           rclad:  cladding cross-section radius
           nclad:  cladding refractive index
           ncore:  core refractive index
           ks:     signal wavenumber.
        """

        if case is None:
            self.L = L             # total fiber length
            self.rcore = rcore     # core cross-section radius
            self.rclad = rclad     # cladding cross-section radius
            self.nclad = nclad     # cladding refractive index
            self.ncore = ncore     # core refractive index
            self.ks = ks           # signal wavenumber
        else:
            self.set(case)

    # BASIC DEFINITIONS:

    def numerical_aperture(self):
        """ Return NA of the fiber """
        # (nclad, ncore) -> NA:
        return sqrt(abs(self.ncore)**2 - abs(self.nclad)**2)

    def wavelength(self):
        """  Return signal wavelength """
        return 2 * pi / self.ks

    def fiberV(self):
        """ Return the V number of the fiber """
        NA = self.numerical_aperture()
        V = NA * self.ks * self.rcore
        return V

    # PROPAGATION CHARACTERISTICS

    def visualize_mode(self, l, m):
        """
        Plot the LP(l,m) mode. Also return the mode as a function of
        scalar coordinates x and y. Note that l and m are both indexed
        to start from 0, so for example, the traditional LP_01 and LP_11
        modes are obtained by calling LP(0, 0) and LP(1, 0), respectively.
        """

        X = self.propagation_constants(l)
        if len(X) < m:
            raise ValueError('For l=%d, only %d fiber modes computed'
                             % (l, len(X)))
        kappa = X[m] / self.rcore
        k0 = self.ks
        beta = sqrt(self.ncore*self.ncore*k0*k0 - kappa*kappa)
        gamma = sqrt(beta*beta - self.nclad*self.nclad*k0*k0)
        Jkrcr = scf.jv(l, kappa*self.rcore)
        Kgrcr = scf.kv(l, gamma*self.rcore)

        def mode(x, y):
            r = sqrt(x*x + y*y)
            theta = atan2(y, x)
            if r < self.rcore:
                u = Kgrcr * scf.jv(l, kappa*r)
            else:
                u = Jkrcr * scf.kv(l, gamma*r)

            u = u*np.cos(l*theta)
            return u

        fig = plt.figure()
        ax = Axes3D(fig)
        lim = self.rcore * 1.5
        X = np.arange(-lim, lim, 2*lim/100)
        Y = np.arange(-lim, lim, 2*lim/100)
        X, Y = np.meshgrid(X, Y)
        vmode = np.vectorize(mode)
        Z = vmode(X, Y)

        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

        return mode

    def propagation_constants(self, l, maxnroots=50):
        """
        Given mode index "l", and fiber parameters "fprms", attempt to
        find all propagation constants by bisection or other nonlinear
        root finders. (Circulalry symmetric modes are obtained with l=0.
        Modes with higher l have angular variations.)
        """

        # If NA is 0, then this is an empty fiber (with V=0) whose
        # propagation constants are given by
        #         beta = sqrt( ks^2 - (jz/rclad)^2 )
        # where jz are roots of l-th Bessel function. Below we reverse
        # engineer X so that this beta is produced by later formulae.

        if abs(self.numerical_aperture()) < 1.e-15:
            jz = scf.jn_zeros(l, maxnroots)
            jz = jz[np.where(jz < self.ks*self.rclad)[0]]
            if len(jz) == 0:
                print('There are no propagating modes for wavenumber ks!')
                print('   ks = ', self.ks)
            # Return X adjusted so that later formulas
            # like kappa = X[m] / rcore   are still valid.
            X = jz * (self.rcore / self.rclad)

            # we are done:
            return X

        # Case of non-empty fibers (V>0):

        V, f1, f2, g1, g2, f, g = self.VJKfun(l)

        # Collect Bessel roots appended with 0 and V:

        jz = scf.jn_zeros(l, maxnroots)
        jz = jz[np.where(jz < V)[0]]
        jz = np.insert(jz, 0, 0)
        jz = np.append(jz, V)
        X = []
        print('\nSEARCHING FOR ROOTS in [0,V] (V = %7g) ' % (V) +
              '-'*29)

        def root_check(x):  # Subfunction used for checking if x is a root
            if abs(f(x)) < 1.e-9 and abs(g(x)) < 1.e-9:
                print('  Root x=%g found with f(x)=%g and g(x)=%g'
                      % (x, f(x), g(x)))
                return True
            else:
                return False

        for i in range(1, len(jz)):

            # Try bisection based on intervals between bessel roots

            a = jz[i-1]+1.e-15
            b = jz[i]-1.e-15
            if np.sign(g(a)*g(b)) < 0:
                print(' SEARCH %1d: Sign change in [%g, %g]' % (i, a, b))
                x = bisect(g, a, b)
                if root_check(x):
                    X += [x]
                else:
                    print('  Bisection on g did not find a root!\n')
                    x = bisect(f, a, b)
                    if root_check(x):
                        X += [x]
                    else:
                        print('  Bisection on f also did not find a root!')

            # Check if end points are roots:

            elif root_check(a):
                X += [a]

            elif root_check(b):
                X += [b]

            # Try nonlinear/gradient-based root finding:

            else:

                x0 = 0.5*(a+b)
                print('  Trying nonlinear solve on g with initial guess %g'
                      % x0)
                x = fsolve(g, x0, xtol=1.e-10)[0]
                if root_check(x):
                    X += [x]
                else:
                    print('  Nonlinear solve on g did not find a root.')
                    print('  Trying nonlinear solve on f with initial guess %g'
                          % x0)
                    x = fsolve(f, x0, xtol=1.e-10)[0]
                    if root_check(x):
                        X += [x]
                    else:
                        print('  Nonlinear solve on f did not find a root.')
                        print('  Giving up on finding roots in [%g, %g].'
                              % (a, b))

        print(' ROOTS FOUND: ', X)
        return X

    def visualize_roots(self, l):
        """
        Visualize two different functions f = f1 - f2 and g = g1 - g2
        whose roots are used to compute propagation constants for
        each mode index "l".
        """

        V, f1, f2, g1, g2, f, g = self.VJKfun(l)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        plt.grid('on')
        plt.rc('text', usetex=True)
        fig.suptitle('Mode index $l=$%1d:' % l +
                     '$\beta$ found by roots of $f$ or $g$',
                     fontsize=14)
        xx = np.arange(0, V, V/500.0)

        axes[0, 0].plot(xx, [f(x) for x in xx])
        axes[0, 0].grid('on')
        axes[0, 0].plot(xx, [0]*len(xx), 'b--')
        axes[0, 0].set_ylim(-V, V)
        axes[0, 0].set_title('$f = f_1 - f_2$')

        axes[1, 0].plot(xx, [f1(x) for x in xx])
        axes[1, 0].grid('on')
        axes[1, 0].plot(xx, [f2(x) for x in xx])
        axes[1, 0].set_ylim(-2*V, 2*V)
        axes[1, 0].legend(['$f_1$', '$f_2$'])

        axes[0, 1].plot(xx, [g(x) for x in xx])
        axes[0, 1].grid('on')
        axes[0, 1].plot(xx, [0]*len(xx), 'b--')
        axes[0, 1].set_ylim(-V, V)
        axes[0, 1].set_title('$g = g_1 - g_2$')

        axes[1, 1].plot(xx, [g1(x) for x in xx])
        axes[1, 1].grid('on')
        axes[1, 1].plot(xx, [g2(x) for x in xx])
        axes[1, 1].set_ylim(-V, V)
        axes[1, 1].legend(['$g_1$', '$g_2$'])

        plt.show(block=False)

    def VJKfun(self, l):
        """
        For the "l"-th mode index of the fiber, return the nonlinear
        functions whose roots give propagation constants.
        """

        V = self.fiberV()
        J = scf.jv
        K = scf.kv

        def jl(X):
            Jlx = J(l, X)
            if abs(Jlx) < 1.e-15:
                if Jlx < 0:
                    Jlx = Jlx - 1e-15
                else:
                    Jlx = Jlx + 1e-15
            return Jlx

        def f1(X):
            JlX = jl(X)
            return J(l+1, X) * X / JlX

        def y(X):
            if X > V:
                Y = 0
            else:
                Y = sqrt(V*V - X*X)
            return Y

        def f2(X):
            Y = y(X)
            return K(l+1, Y) * Y / K(l, Y)

        def f(X):  # Propagation constant is a root of f(X)
            return f1(X) - f2(X)

        def g1(X):
            JlX = jl(X)
            Y = y(X)
            return J(l+1, X) * K(l, Y) / (JlX * K(l+1, Y))

        def g2(X):
            return y(X)/max(X, 1e-15)

        def g(X):  # Propagation constant is also a root of g(X)
            return g1(X) - g2(X)

        return V, f1, f2, g1, g2, f, g

    # PREPROGRAMMED FIBER CASES:

    def set(self, case):

        if case == 'artificial_singlemode':

            L = 0.1
            rcore = 1
            rclad = 10
            k0 = 4
            NA = 0.5
            ncore = 1
            nclad = sqrt(ncore*ncore - NA*NA)

        elif case == 'artificial_multimode':

            L = 10
            rcore = 1
            rclad = 10
            k0 = 20
            NA = 0.5
            ncore = 1
            nclad = sqrt(ncore*ncore - NA*NA)

        elif case == 'Nufern':

            # ** 25/400 Ytterbium-Doped LMA Double Clad Fiber **
            # From Spec sheet provided by Grosek:

            rcore = 1.25e-5
            rclad = 2e-4
            wavelen = 1.064e-6
            ncore = 1.450971
            k0 = 2 * pi / wavelen
            NA = 0.06
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation

        elif case == 'Thulium':

            # ** (25/400?) Thulium-Doped (LMA Double Clad Fiber?) **
            # From Spec sheet provided by Grosek:

            rcore = 1.25e-5         # assumed same as Nufern
            rclad = 2e-4            # assumed same as Nufern
            wavelen = 2.110e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.06               # assumed same as Nufern
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation ?

        elif case == 'book':
            L = 0.1
            rcore = 3e-6
            rclad = 1e-3
            wavelen = 1.55e-6
            k0 = 2 * pi / wavelen
            nclad = 1.460
            ncore = 1.469

        elif case == 'empty':
            L = 10
            rcore = 0.25
            rclad = 1
            k0 = 100
            nclad = 1.0
            ncore = 1.0

        else:
            raise ValueError('Unknown fiber parameter case %s' % case)

        self.__init__(None, L=L, rcore=rcore, rclad=rclad,
                      nclad=nclad, ncore=ncore, ks=k0)

    def print_params(self):
        print('\nFIBER PARAMETERS: ' + '-'*54)
        print('ks:         %20g' % self.ks +
              '{:>40}'.format('signal frequency'))
        print('wavelength: %20g' % (2*pi/self.ks) +
              '{:>40}'.format('signal wavelength'))
        print('L:          %20g' % (self.L) +
              '{:>40}'.format('fiber length'))
        print('V:          %20g' % (self.fiberV()) +
              '{:>40}'.format('V-number of fiber'))
        if isinstance(self.ncore, complex):
            print('ncore:      '
                  '{:>10f}'.format(self.ncore.real) +
                  '{:<+9f}j'.format(self.ncore.imag) +
                  '{:>40}'.format('core refractive index'))
        else:
            print('ncore:      %20g' % (self.ncore) +
                  '{:>40}'.format('core refractive index'))
        if isinstance(self.nclad, complex):
            print('nclad:      '
                  '{:>10f}'.format(self.nclad.real) +
                  '{:<+9f}j'.format(self.nclad.imag) +
                  '{:>40}'.format('cladding refractive index'))
        else:
            print('nclad:      %20g' % (self.nclad) +
                  '{:>40}'.format('cladding refractive index'))
        print('rcore:      %20g' % (self.rcore) +
              '{:>40}'.format('core radius'))
        print('rclad:      %20g' % (self.rclad) +
              '{:>40}'.format('cladding radius'))
        print('-'*72)
