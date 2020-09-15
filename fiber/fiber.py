"""
This class models STEP-INDEX optical fibers.  Methods are provided
to calculate propagation constants, fiber's transverse guided modes,
and leaky modes/resonances using (semi)ANALYTIC calculations.

(Numerical mode computation routines using feast are in a different
class FiberModeNonDim.)
"""

from math import pi, sqrt, atan2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import fsolve, bisect
from scipy.special import hankel1, jv, kv, jn_zeros
from cxroots import Rectangle
import sympy as sm


class Fiber:

    """ A class of step-index cylindrical fibers """

    def __init__(self, case=None,
                 L=None, rcore=None, rclad=None, nclad=None, ncore=None,
                 ks=None, extra_wavelens=None):
        """
        A step-index fiber object F can be made using a preprogrammed case
        name (such as 'Nufern_Yb', 'liekki_1', 'corning_smf_28_1', etc), e.g.,
           F = Fiber('Nufern_Yb').
        Alternately, a fiber object can be made by providing these kwargs:
           L:      total fiber length
           rcore:  core cross-section radius
           rclad:  cladding cross-section radius
           nclad:  cladding refractive index
           ncore:  core refractive index
           ks:     signal wavenumber.
           extra_wavelens: tone and/or ASE wavelengths
        """

        if case is None:
            self.L = L             # total fiber length
            self.rcore = rcore     # core cross-section radius
            self.rclad = rclad     # cladding cross-section radius
            self.nclad = nclad     # cladding refractive index
            self.ncore = ncore     # core refractive index
            self.ks = ks           # signal wavenumber
            if extra_wavelens:
                # tone and/or ase wavenumbers
                self.ke = [2*pi/lam for lam in extra_wavelens]
        else:
            self.set(case)

    # BASIC DEFINITIONS:

    def numerical_aperture(self):
        """ Return NA of the fiber """
        # (nclad, ncore) -> NA:
        return sqrt(abs(self.ncore)**2 - abs(self.nclad)**2)

    def wavelength(self, tone=False):
        """  Return signal wavelength """
        if tone:
            return [2 * pi / self.ks] + [2 * pi / k for k in self.ke]
        else:
            return 2 * pi / self.ks

    def fiberV(self, tone=False):
        """
        Returns the V number of the fiber.
        If multitone, returns a list 'V' where
        V[0] is the V-number coresponding to signal
        and tone V-numbers afterwards.
        """
        NA = self.numerical_aperture()
        V = NA * self.ks * self.rcore
        if tone:
            V = [V] + [NA * kt * self.rcore for kt in self.ke]
        return V

    # PROPAGATION CHARACTERISTICS

    def XtoBeta(self, X, v=None):
        """
        Convert nondimensionalized roots X to actual propagation
        constants Beta of guided modes.
        """
        V = self.fiberV() if v is None else v
        a = self.rcore
        ks = V / (self.numerical_aperture() * a)
        kappas = [x/a for x in X]
        betas = [sqrt(self.ncore*self.ncore*ks*ks - kappa*kappa)
                 for kappa in kappas]
        return betas

    def ZtoBeta(self, Z, v=None):
        """
        Convert nondimensional Z in the complex plan to complex propagation
        constants Beta of leaky modes.
        """
        V = self.fiberV() if v is None else v
        a = self.rcore
        ks = V / (self.numerical_aperture() * a)
        return np.sqrt((ks*self.ncore)**2-(Z/a)**2)

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
        Jkrcr = jv(l, kappa*self.rcore)
        Kgrcr = kv(l, gamma*self.rcore)

        def mode(x, y):
            r = sqrt(x*x + y*y)
            theta = atan2(y, x)
            if r < self.rcore:
                u = Kgrcr * jv(l, kappa*r)
            else:
                u = Jkrcr * kv(l, gamma*r)

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

    def propagation_constants(self, l, maxnroots=50, v=None):
        """
        Given mode index "l", attempt to find all propagation constants by
        bisection or other nonlinear root finders. (Circularly
        symmetric modes are obtained with l=0.  Modes with higher l
        have angular variations.)
        """
        if v is None:
            ks = self.ks
            V = self.fiberV()
        else:
            ks = v / (self.numerical_aperture() * self.rcore)
            V = v

        # Case of empty fibers:

        if abs(self.numerical_aperture()) < 1.e-15:

            # If NA is 0, then this is an empty fiber (with V=0) whose
            # propagation constants are given by
            #         beta = sqrt( ks^2 - (jz/rclad)^2 )
            # where jz are roots of l-th Bessel function. Below we reverse
            # engineer X so that this beta is produced by later formulae.

            jz = jn_zeros(l, maxnroots)
            jz = jz[np.where(jz < ks*self.rclad)[0]]
            if len(jz) == 0:
                print('There are no propagating modes for wavenumber ks!')
                print('   ks = ', ks)
            # Return X adjusted so that later formulas
            # like kappa = X[m] / rcore   are still valid.
            X = jz * (self.rcore / self.rclad)

            # we are done:
            return X

        # Case of non-empty fibers (V>0):

        _, f1, f2, g1, g2, f, g = self.VJKfun(l, v=V)

        # Collect Bessel roots appended with 0 and V:

        jz = jn_zeros(l, maxnroots)
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
        plt.grid(True)
        plt.rc('text', usetex=True)
        fig.suptitle(r'Mode index $l=$%1d:' % l +
                     r'$\beta$ found by roots of $f$ or $g$',
                     fontsize=14)
        xx = np.arange(0, V, V/500.0)

        axes[0, 0].plot(xx, [f(x) for x in xx])
        axes[0, 0].grid(True)
        axes[0, 0].plot(xx, [0]*len(xx), 'b--')
        axes[0, 0].set_ylim(-V, V)
        axes[0, 0].set_title('$f = f_1 - f_2$')

        axes[1, 0].plot(xx, [f1(x) for x in xx])
        axes[1, 0].grid(True)
        axes[1, 0].plot(xx, [f2(x) for x in xx])
        axes[1, 0].set_ylim(-2*V, 2*V)
        axes[1, 0].legend(['$f_1$', '$f_2$'])

        axes[0, 1].plot(xx, [g(x) for x in xx])
        axes[0, 1].grid(True)
        axes[0, 1].plot(xx, [0]*len(xx), 'b--')
        axes[0, 1].set_ylim(-V, V)
        axes[0, 1].set_title('$g = g_1 - g_2$')

        axes[1, 1].plot(xx, [g1(x) for x in xx])
        axes[1, 1].grid(True)
        axes[1, 1].plot(xx, [g2(x) for x in xx])
        axes[1, 1].set_ylim(-V, V)
        axes[1, 1].legend(['$g_1$', '$g_2$'])

        plt.ion()
        plt.show()
        plt.show(block=False)

    def VJKfun(self, l, v=None):
        """
        For the "l"-th mode index of the fiber, return the nonlinear
        functions whose roots give propagation constants.
        v: V-number of the fiber
        """

        V = self.fiberV() if v is None else v
        J = jv
        K = kv

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

    def VJHfuns(self, l):
        """
        For the "l"-th mode index, return a nonlinear function of a
        nondimensional variable Z whose nondimensionalized roots give
        leaky modes.  The function is returned as a string (with letter Z)
        which can be evaluated for specific Z values later.
        """
        z, nu = sm.symbols('z nu')
        V = self.fiberV()
        x = sm.sqrt(V*V + z*z)
        g = z*sm.besselj(l, x)*sm.hankel1(l+1, z) - \
            x*sm.besselj(l+1, x)*sm.hankel1(l, z)
        dg = g.diff(z).expand()

        dgstr = str(dg).replace('z', 'Z').     \
            replace('besselj', 'jv').          \
            replace('nu', 'l').                \
            replace('sqrt', 'np.sqrt')
        gstr = str(g).replace('z', 'Z').       \
            replace('besselj', 'jv').          \
            replace('nu', 'l').                \
            replace('sqrt', 'np.sqrt')
        return gstr, dgstr

    def leaky_propagation_constants(self, l, xran=None, yran=None):
        """
        Given a mode index "l" indicating angular variation (the
        radially symmetric case being l=0), search the following
        rectangular region (given by tuples xran, yran)
              [xran[0], xran[1]]  x  [yran[0], yran[1]]
        of the complex plane for roots that yield leaky outgoing modes.
        (Be warned that the root finder is not as robust as the real
        line root searching algorithm for guided modes.)
        """

        if xran is None:
            xran = (0.01, 3*self.fiberV())
        if yran is None:
            yran = (-2, 0)
        print('Searching region (%g, %g) x (%g, %g) in complex plane'
              % (xran[0], xran[1], yran[0], yran[1]))
        gstr, dgstr = self.VJHfuns(l)
        try:
            rect = Rectangle(xran, yran)
            r = rect.roots(lambda Z: eval(gstr),
                           lambda Z: eval(dgstr),
                           rootErrTol=1.e-13, newtonStepTol=1.e-15)
        except RuntimeError as err:
            print('Root search failed:\n', err.__str__())
            dx = xran[1]-xran[0]
            dy = yran[1]-yran[0]
            xran2 = (xran[0] + 0.01 * dx, xran[1] - 0.01 * dx)
            yran2 = (yran[0] + 0.01 * dy, yran[1] - 0.01 * dy)
            print('Retrying in adjusted search region (%g, %g) x (%g, %g)'
                  % (xran2[0], xran2[1], yran2[0], yran2[1]))
            r = self.leaky_propagation_constants(l, xran=xran2, yran=yran2)
        return r.roots

    def visualize_leaky_mode(self, Z, l, corelim=2):
        """
        Given a complex propagation constant Z obtained from
        self.leaky_propagation_constants(l),  compute the corresponding
        leaky mode. Return its values F at a meshgrid of points (X, Y) and
        plot it. If "corelim" is given, this grid discretizes the xy region
        [-lim, lim] x [-lim, lim] where lim is corelim x core radius.
        """

        V = self.fiberV()
        X = np.sqrt(Z*Z+V*V)
        a = self.rcore
        alpha0 = Z/a
        alpha1 = X/a
        B = jv(l, X)
        A = hankel1(l, Z)

        def modefun(x, y):
            r = np.sqrt(x*x + y*y)
            theta = atan2(y, x)
            if r < a:
                u = A * jv(l, alpha1*r)
            else:
                u = B * hankel1(l, alpha0*r)
            u = u*np.cos(l*theta)
            return u

        lim = a * corelim
        X = np.arange(-lim, lim, 2*lim/300)
        Y = np.arange(-lim, lim, 2*lim/300)
        X, Y = np.meshgrid(X, Y)
        vmode = np.vectorize(modefun)
        F = vmode(X, Y)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.contour3D(X, Y, F.real, 30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Real part of the mode)')
        plt.show(block=False)

        return X, Y, F, modefun, ax

    # PREPROGRAMMED FIBER CASES & OTHER UTILITIES:

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
            rclad = 5
            k0 = 100
            NA = 0.05
            ncore = 1
            nclad = sqrt(ncore*ncore - NA*NA)

        elif case == 'Nufern_Yb':

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

        elif case == 'Nufern_Tm':

            # ** (25/400) Thulium-Doped LMA Double Clad Fiber **
            # From Spec sheet provided by Grosek:

            rcore = 1.25e-5
            rclad = 2e-4
            wavelen = 2.110e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.1
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation

        elif case == 'Nufern_Tm2':

            # ** (25/400) Thulium-Doped LMA Double Clad Fiber **
            # From Spec sheet provided by Grosek:

            rcore = 1.25e-5
            rclad = 2e-4
            wavelen = 2.133e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.1
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation

        elif case == 'Nufern_Tm3':

            # ** (25/400) Thulium-Doped LMA Double Clad Fiber **
            # From Spec sheet provided by Grosek:

            rcore = 1.25e-5
            rclad = 2e-4
            wavelen = 2.170e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.1
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation

        elif case == 'Toned_Tm':

            rcore = 1e-5
            rclad = 2e-4
            # specify signal wavelength first,
            # then specify tones
            wavelen = 2.110e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.099
            nclad = sqrt(ncore*ncore - NA*NA)
            tone_wavelens = [1.9305876e-6]

            extra_wavelens = tone_wavelens

            L = 0.1  # to be varied for each simulation

        elif case == 'Toned_Tm_test':

            rcore = 1e-5
            rclad = 2.75e-4
            # specify signal wavelength first,
            # then specify tones
            wavelen = 2.133e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.09
            nclad = sqrt(ncore*ncore - NA*NA)
            tone_wavelens = [1.95e-6]

            extra_wavelens = tone_wavelens

            L = 0.1  # to be varied for each simulation

        elif case == 'TonedASE_Tm':

            rcore = 1e-5
            rfiber = 2.75e-4
            rclad = rfiber
            # specify signal wavelength first,
            # then specify tones
            wavelen = 2.11e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.09
            nclad = sqrt(ncore*ncore - NA*NA)
            tone_wavelens = [1.9306356965146654e-6]
            ASE_wavelens = [1.925e-6, 1.975e-6, 2.025e-6, 2.075e-6, 2.125e-6]

            extra_wavelens = tone_wavelens + ASE_wavelens
            L = 0.1  # to be varied for each simulation

        elif case == 'TonedASE_Tm2':

            rcore = 1e-5
            rfiber = 2.75e-4
            rclad = rfiber
            # specify signal wavelength first,
            # then specify tones
            wavelen = 2.133e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.09
            nclad = sqrt(ncore*ncore - NA*NA)
            tone_wavelens = [1.9498737599293853e-6]
            ASE_wavelens = [1.925e-6, 1.975e-6, 2.025e-6, 2.075e-6, 2.125e-6]

            extra_wavelens = tone_wavelens + ASE_wavelens
            L = 0.1  # to be varied for each simulation

        elif case == 'TonedASE_Tm3':

            rcore = 1e-5
            rfiber = 2.75e-4
            rclad = rfiber
            # specify signal wavelength first,
            # then specify tones
            wavelen = 2.170e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.09
            nclad = sqrt(ncore*ncore - NA*NA)
            tone_wavelens = [1.9807473196535323e-6]
            ASE_wavelens = [1.925e-6, 1.975e-6, 2.025e-6, 2.075e-6, 2.125e-6]

            extra_wavelens = tone_wavelens + ASE_wavelens
            L = 0.1  # to be varied for each simulation

        elif case == 'Toned_Yb':
            rcore = 1.25e-5
            rclad = 2e-4
            ncore = 1.450971
            NA = 0.06
            nclad = sqrt(ncore*ncore - NA*NA)
            L = 0.1   # to be varied for each simulation
            # specify signal wavelength first,
            # then specify tones
            wavelen = 1.064e-6
            k0 = 2 * pi / wavelen
            tone_wavelens = [1.040e-6]

            extra_wavelens = tone_wavelens

        elif case == 'LLMA_Yb':

            # ** Variation of Nufern Ytterbium-Doped LMA Double Clad Fiber **
            # with large rcore

            rcore = 3.7e-5
            rclad = 2e-4
            wavelen = 1.064e-6
            ncore = 1.450971
            k0 = 2 * pi / wavelen
            NA = 0.06
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation

        elif case == 'LLMA_Tm':

            # ** Variation of Nufern Thulium-Doped LMA Double Clad Fiber **
            # with large rcore

            rcore = 3.7e-5
            rclad = 2e-4
            wavelen = 2.110e-6
            ncore = 1.439994
            k0 = 2 * pi / wavelen
            NA = 0.1
            nclad = sqrt(ncore*ncore - NA*NA)

            L = 0.1   # to be varied for each simulation

        elif case == 'schermer_cole':

            rcore = 1.25e-5
            rclad = 16 * rcore
            wavelen = 1.064e-6
            nclad = 1.52
            NA = 0.1
            ncore = sqrt(NA*NA + nclad*nclad)
            k0 = 2 * pi / wavelen

            L = 0.1

        elif case == 'corning_smf_28_1':

            # Single mode fiber from Schermer and Cole paper,
            # as well as BYU specs document from April 2002.
            L = 0.1
            rcore = 4.1e-6
            rclad = 125e-6
            wavelen = 1.320e-6
            k0 = 2 * pi / wavelen
            NA = 0.117
            nclad = 1.447
            ncore = sqrt(nclad*nclad + NA*NA)

        elif case == 'corning_smf_28_2':

            # Single mode fiber from Schermer and Cole paper,
            # as well as BYU specs document from April 2002.
            L = 0.1
            rcore = 4.1e-6
            rclad = 125e-6
            wavelen = 1.550e-6
            k0 = 2 * pi / wavelen
            NA = 0.117
            nclad = 1.440
            ncore = sqrt(nclad*nclad + NA*NA)

        elif case == 'liekki_1':

            # Multi-mode mode fiber from Schermer and Cole paper,
            # with cladding radius obtained from nLight spec sheet
            # on Liekki passive 25/250DC fibers (slightly different
            # specs than the specified 25/240DC in Schermer and Cole).
            L = 0.1
            rcore = 1.25e-5
            rclad = 1.25e-4
            wavelen = 6.33e-7
            k0 = 2 * pi / wavelen
            NA = 0.06
            nclad = 1.46
            ncore = sqrt(nclad*nclad + NA*NA)

        elif case == 'liekki_2':

            # Multi-mode mode fiber from Schermer and Cole paper,
            # with cladding radius obtained from nLight spec sheet
            # on Liekki passive 25/250DC fibers (slightly different
            # specs than the specified 25/240DC in Schermer and Cole).
            L = 0.1
            rcore = 1.25e-5
            rclad = 1.25e-4
            wavelen = 8.30e-7
            k0 = 2 * pi / wavelen
            NA = 0.06
            nclad = 1.46
            ncore = sqrt(nclad*nclad + NA*NA)

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

        elif case == 'empty_2':
            L = 10
            rcore = 0.5
            rclad = 1
            k0 = 100
            nclad = 1.0
            ncore = 1.0

        else:
            raise ValueError('Unknown fiber parameter case %s' % case)

        if case.startswith('Toned'):
            self.__init__(None, L=L, rcore=rcore, rclad=rclad,
                          nclad=nclad, ncore=ncore, ks=k0,
                          extra_wavelens=extra_wavelens)
        else:
            self.__init__(None, L=L, rcore=rcore, rclad=rclad,
                          nclad=nclad, ncore=ncore, ks=k0,
                          extra_wavelens=None)

    def print_params(self):
        print('\nFIBER PARAMETERS: ' + '-'*54)
        print('ks:         %20g' % self.ks +
              '{:>40}'.format('signal wavenumber'))
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
        if 'ke' in self.__dir__():
            print('ke:         {}'.format(self.ke) +
                  '{:>40}'.format('tone wave numbers'))
            print('Ve:         {}'.format(self.fiberV(tone=True)) +
                  '{}'.format('tone V-numbers'))
        print('-'*72)
