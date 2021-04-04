import ngsolve as ng
from ngsolve import grad, dx
import numpy as np
from fiberamp.fiber.spectralprojpoly import SpectralProjNGPoly
from pyeigfeast.spectralproj.ngs import NGvecs, SpectralProjNGGeneral


class ModeSolver:

    """This class contains algorithms to compute modes of various fibers.

    METHOD FOR SCALAR MODES

    The Helmholtz mode in physical coordinates is given by

         Δu + k² n² u = β² u.

    The transverse refractive index n is a function implemented
    by a derived class, and it's assumed that it takes a
    constant value n₀ outside a fixed radius.

    What's implemented are algorithms for a non-dimensional version
    of the above, obtained after fixing a characteristic length scale L,
    and transforming the above to the following

         -Δu + V u = Z² u

    where Z² = L² (k² n₀² - β²).   Here the nondimensional function V is
    an index well, akin to a Schrödinger potential well, and is given in
    terms of the physical material properties by

          V = L²k² (n₀² - n²)   if r < R₀,
          V = 0                 if r > R₀.

    Here R₀ is the nondimensional radius such that n is constant
    beyond L R₀ in the  physical domain.

    * R = radius marking the start of PML whenever PML is
      used (for leaky modes). Note that R > R₀.

    * Rout = final outer radius terminating the computational domain.
      The circle r = Rout is assumed to be a boundary region
      named 'OuterCircle' of  the given mesh.

    * When PML is used, it is put in the region R < r < Rout.
      The PML region R < r < Rout is assumed to be called 'Outer'
      in the given mesh.

    """

    def __init__(self, mesh, L, k, n0):

        self.mesh = mesh
        self.L = L
        self.k = k
        self.n0 = n0
        self.ngspmlset = False

        print('ModeSolver: Checking if mesh has required regions')

        if sum(self.mesh.Materials('Outer').Mask()) == 0:
            raise ValueError('Input mesh must have a region called Outer')
        if sum(self.mesh.Boundaries('OuterCircle').Mask()) == 0:
            raise ValueError('Input mesh must have a boundary ' +
                             'called OuterCircle')

    def betafrom(self, Z2):
        """
        Returns physical propagation constants (β), given
        nondimensional Z² values, input in Z2, per the formula
        β = sqrt(L²k²n₀² - Z²) / L . """

        return np.sqrt((self.L*self.k*self.n0)**2 - Z2) / self.L

    # ###################################################################
    # FREQ-DEPENDENT PML BY POLYNOMIAL EIGENPROBLEM #####################

    def polypmlsystem(self, p, alpha=1):
        """
        Returns AA, X

          AA is a list of 4 cubic matrix polynomial coefficients on FE space X
        """

        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with poly.')

        dx_pml = dx(definedon=self.mesh.Materials('Outer'))
        dx_int = dx(definedon=~self.mesh.Materials('Outer'))
        R = self.R
        s = 1 + 1j * alpha
        x = ng.x
        y = ng.y
        r = ng.sqrt(x*x+y*y) + 0j
        X = ng.H1(self.mesh, order=p, complex=True)
        u, v = X.TnT()
        ux, uy = grad(u)
        vx, vy = grad(v)

        AA = [ng.BilinearForm(X, check_unused=False)]
        AA[0] += (s*r/R) * grad(u) * grad(v) * dx_pml
        AA[0] += s * (r-R)/(R*r*r) * (x*ux+y*uy) * v * dx_pml
        AA[0] += s * (R-2*r)/r**3 * (x*ux+y*uy) * (x*vx+y*vy) * dx_pml
        AA[0] += -s**3 * (r-R)**2/(R*r) * u * v * dx_pml

        AA += [ng.BilinearForm(X)]
        AA[1] += grad(u) * grad(v) * dx_int
        AA[1] += self.V * u * v * dx_int
        AA[1] += 2 * (r-R)/r**3 * (x*ux+y*uy) * (x*vx+y*vy) * dx_pml
        AA[1] += 1/r**2 * (x*ux+y*uy) * v * dx_pml
        AA[1] += -2*s*s*(r-R)/r * u * v * dx_pml

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[2] += R/s/r**3 * (x*ux+y*uy) * (x*vx+y*vy) * dx_pml
        AA[2] += -R*s/r * u * v * dx_pml

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[3] += -u * v * dx_int

        with ng.TaskManager():
            for i in range(len(AA)):
                AA[i].Assemble()

        return AA, X

    def leakymode(self, p, ctr=2, rad=0.1, alpha=1, npts=8,
                  within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
                  nspan=5, seed=1, inverse=None, **feastkwargs):
        """
        Solve the polynomial PML eigenproblem to compute leaky modes with
        losses [Nannen+Wess]. A custom polynomial feast uses the given
        centers and radii to search for the modes.

        PARAMETERS:

        p:        Polynomial degree of finite elements.
        alpha:    PML strength.
        nspan:    Dimension of random initial eigenspace iterate.
        seed:     Fix seed for reproducing random initial iterate.
        npts, ctrs, radi, within, rhoinv, quadrule:
                  These paramaters are passed to SpectralProjNGPoly
                  constructor. See documentation there.
        feastkwargs: Further keyword arguments passed to the feast(...)
                  method of the spectral projector. See documentation there.

        OUTPUTS:  Z, y, yl, beta, P, moreoutputs

        Z: nondimensional polynomial eigenvalue
        y: right eigenspan
        yl: left eigenspan
        beta: physical propagation constant
        P: the SpectralProjNGPoly object used to compute Z
        moreoutputs: dictionary of more outputs

        """

        print('ModeSolver.leakymode called on object with these settings:\n',
              self)

        AA, X = self.polypmlsystem(p=p, alpha=alpha)
        X3 = ng.FESpace([X, X, X])
        print('Set freq-dependent PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout-self.R))

        Y = NGvecs(X3, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)

        P = SpectralProjNGPoly(AA, X, radius=rad, center=ctr, npts=npts,
                               within=within, rhoinv=rhoinv,
                               quadrule=quadrule, inverse=inverse)

        Z, Y, hist, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                 **feastkwargs)
        ews, cgd = hist[-2], hist[-1]
        if not cgd:
            print('*** Iterations did not converge')

        y = P.first(Y)
        yl = P.last(Yl)
        y.centernormalize(self.mesh(0, 0))
        yl.centernormalize(self.mesh(0, 0))

        print('Results:\n Z:', Z)
        beta = self.betafrom(Z**2)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        def outint(u):
            out = self.mesh.Boundaries('OuterCircle')
            s = ng.Integrate(u*ng.Conj(u), out, ng.BND).real
            return ng.sqrt(s)

        bdrnrm = y.applyfnl(outint)
        print('Actual boundary norm = %.1e' % max(bdrnrm))
        if np.max(bdrnrm) > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')
            decayrate = alpha * (self.Rout - self.R) + \
                self.R * Z.imag
            bdryval = np.exp(-decayrate) / np.sqrt(np.abs(Z)*np.pi/2)
            bdrnrm0 = bdryval*2*np.pi*self.Rout
            print('PML decay estimates boundary norm ~ %.1e'
                  % max(bdrnrm0))

        moreoutputs = {'longY': Y, 'longYl': Yl,
                       'ewshistory': ews, 'bdrnorm': bdrnrm}

        return Z, y, yl, beta, P, moreoutputs

    # ###################################################################
    # NGSOLVE AUTOMATIC PML #############################################

    def autopmlsystem(self, p, alpha=1):
        """
        Set up PML using NGSolve's automatic PML using in-built
        mesh transformations.
        """
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        self.ngspmlset = True

        radial = ng.pml.Radial(rad=self.R,
                               alpha=alpha*1j, origin=(0, 0))
        self.mesh.SetPML(radial, 'Outer')
        print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout-self.R))
        X = ng.H1(self.mesh, order=p, complex=True)

        u, v = X.TnT()
        a = ng.BilinearForm(X)
        b = ng.BilinearForm(X)
        a += (grad(u) * grad(v) + self.V * u * v) * dx
        b += u * v * dx
        with ng.TaskManager():
            a.Assemble()
            b.Assemble()

        return a, b, X

    def leakymode_auto(self, p, radiusZ2=0.1, centerZ2=4,
                       alpha=1, npts=8, nspan=5, seed=1,
                       inverse='umfpack', **feastkwargs):
        """
        Compute leaky modes by solving a linear eigenproblem using
        the frequency-independent automatic PML mesh map of NGSolve
        and using non-selfadjoint FEAST.

        INPUTS:

        * radiusZ2, centerZ2:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * Remaining inputs are the as documented in leakymode(..).

        OUTPUTS:   zsqr, Yl, Y, P

        * zsqr: computed resonance values Z²
        * Yl, Y: left and right eigenspans
        * P: SpectralProjNGGeneral object that computed Y, Yl
        """

        print('ModeSolver.leakymode called on object with these settings:\n',
              self)
        a, b, X = self.autopmlsystem(p, alpha=alpha)

        P = SpectralProjNGGeneral(X, a.mat, b.mat,
                                  radiusZ2, centerZ2, npts,
                                  inverse=inverse)
        Y = NGvecs(X, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        zsqr, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                       **feastkwargs)
        beta = self.betafrom(zsqr)

        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        return zsqr, Yl, Y, beta, P
