import ngsolve as ng
from ngsolve import curl, grad, dx
import numpy as np
import sympy as sm
from pyeigfeast import NGvecs, SpectralProjNGGeneral, SpectralProjNG
from pyeigfeast import SpectralProjNGR, SpectralProjNGPoly


class ModeSolver:

    """This class contains algorithms to compute modes of various fibers,
    including MICROSTRUCTURED fibers with or without radial symmetry.
    The key inputs are cross section mesh, a characteristic length L,
    the constant refractive index n0 in the unbounded complement, the
    refractive index and the nondimensional index well V (all
    described below in more detail). The latter two (index & V) are
    expected to be provided as attributes of derived classes
    containing configuration details of specific fibers.

    HOW SCALAR MODES ARE COMPUTED:

    The Helmholtz mode in physical coordinates is given by

         Δu + k² n² u = β² u.

    The transverse refractive index n is a function implemented
    by a derived class, and it's assumed that it takes a
    constant value n₀ outside a fixed radius.

    What's implemented are algorithms for a non-dimensional version
    of the above, obtained after fixing a characteristic length scale L,
    and transforming the above to the following

         -Δu + V u = Z² u

    where Z² = L² (k² n₀² - β²) and the mode u (not to be confused with the
    physical mode) is defined on a non-dimensional (unit sized) domain.
    The nondimensional function V is an index well, akin  to a
    Schrödinger potential well, and is given in terms of the
    physical material properties by

          V = L²k² (n₀² - n²)   if r < R₀,
          V = 0                 if r > R₀.

    Here R₀ is the nondimensional radius such that n is constant
    beyond LR₀ in the  physical domain.

    CLASS ATTRIBUTES:

    * self.L, self.n0: represent L, n₀ as described above.
    * self.mesh: input mesh of non-dimensionalized transverse domain.

    Attributes assumed to be set by derived classes:

    * self.R = radius marking the start of PML whenever PML is
      used (for leaky modes). Note that R > R₀.

    * self.Rout = final outer radius terminating the computational domain.
      The circle r = Rout is assumed to be a boundary region
      named 'OuterCircle' of  the given mesh.

    * When PML is used, it is put in the region R < r < Rout.
      The PML region R < r < Rout is assumed to be called 'Outer'
      in the given mesh.

    * self.V and self.index represent the L-nondimensionalized index
      well function V, and the physical refractive index profile of
      the fiber, both of which must be set by a derived class before
      calling any of the implemented algorithms.

    * self.k represents the wavenumber k in the definition of V, which
      must be set by (and can be changed by) a derived class before calling
      any of the implemented algorithms.

    """

    def __init__(self, mesh, L, n0):

        self.mesh = mesh
        self.L = L
        self.n0 = n0
        self.ngspmlset = False

        print('ModeSolver: Checking if mesh has required regions')
        print('Mesh has ', mesh.ne, ' elements, ', mesh.nv, ' points, '
              ' and ', mesh.nedge, ' edges.')
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

    def sqrZfrom(self, betas):
        """ Return values of nondimensional Z squared, given physical
        propagation constants betas, ie, return Z² = L² (k²n₀² - β²). """
        return (self.L*self.k*self.n0)**2 - (self.L*betas)**2

    def boundarynorm(self, y):
        """
        Returns  L² norm of all functions in the span y restricted to
        the outermost boundary r= Rout.
        """

        def outint(u):
            out = self.mesh.Boundaries('OuterCircle')
            s = ng.Integrate(u*ng.Conj(u), out, ng.BND).real
            return ng.sqrt(s)
        bdrnrms = y.applyfnl(outint)
        print('Mode boundary L² norm = %.1e' % np.max(bdrnrms))
        return bdrnrms

    def estimatepolypmldecay(self, Z, alpha):
        """
        Returns an estimate of mode boundary norm, per predicted decay of
        the frequency dependent PML for given Z and alpha.
        """

        decayrate = alpha * (self.Rout - self.R) + \
            self.R * Z.imag
        bdryval = np.exp(-decayrate) / np.sqrt(np.abs(Z)*np.pi/2)
        bdrnrm0 = bdryval*2*np.pi*self.Rout
        print('PML decay estimates boundary norm ~ %.1e'
              % max(bdrnrm0))
        return bdrnrm0

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
                try:
                    AA[i].Assemble()
                except Exception:
                    print('*** Trying again with larger heap')
                    ng.SetHeapSize(int(1e9))
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

        bdrnrm = self.boundarynorm(y)
        if np.max(bdrnrm) > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')
            self.estimatepolypmldecay(Z, alpha)

        moreoutputs = {'longY': Y, 'longYl': Yl,
                       'ewshistory': ews, 'bdrnorm': bdrnrm,
                       'converged': cgd}

        return Z, y, yl, beta, P, moreoutputs

    def leakymode_poly(self, p, ctr=2, rad=0.1, alpha=1, npts=8,
                       within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
                       nspan=5, seed=1, inverse=None, **feastkwargs):
        """
        This method is an alternate implementation of the polynomial
        eigensolver using NGSolve bilinear forms in a product finite
        element space. It has been useful sometimes in testing and
        debugging. It should give the same results as leakymode(...),
        and its arguments are as documented in leakymode(...).
        It's more expensive than leakymode(...).
        """

        print('ModeSolver.leakymode_poly called on this object:\n',
              self)
        print('Set freq-dependent PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout-self.R))
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with poly.')

        X = ng.H1(self.mesh, order=p, complex=True)

        # This implementation of [Nannen+Wess]'s frequency-dependent PML is
        # makes a cubic eigenproblem using 3 copies of X:

        X3 = ng.FESpace([X, X, X])

        u0, u1, u2 = X3.TrialFunction()
        v0, v1, v2 = X3.TestFunction()
        u0x, u0y = grad(u0)
        u1x, u1y = grad(u1)
        u2x, u2y = grad(u2)
        v2x, v2y = grad(v2)

        pmlbegin = self.R
        dx_pml = dx(definedon=self.mesh.Materials('Outer'))
        dx_int = dx(definedon=self.mesh.Materials('core|clad'))

        R = pmlbegin
        s = 1 + 1j * alpha
        x = ng.x
        y = ng.y
        r = ng.sqrt(x*x+y*y) + 0j

        A = ng.BilinearForm(X3)
        B = ng.BilinearForm(X3)

        A += u1 * v0 * dx
        A += u2 * v1 * dx

        A += (s*r/R) * grad(u0) * grad(v2) * dx_pml
        A += s * (r-R)/(R*r*r) * (x*u0x+y*u0y) * v2 * dx_pml
        A += s * (R-2*r)/r**3 * (x*u0x+y*u0y) * (x*v2x+y*v2y) * dx_pml
        A += -s**3 * (r-R)**2/(R*r) * u0 * v2 * dx_pml

        A += grad(u1) * grad(v2) * dx_int
        A += self.V * u1 * v2 * dx_int
        A += 2 * (r-R)/r**3 * (x*u1x+y*u1y) * (x*v2x+y*v2y) * dx_pml
        A += 1/r**2 * (x*u1x+y*u1y) * v2 * dx_pml
        A += -2*s*s*(r-R)/r * u1 * v2 * dx_pml

        A += R/s/r**3 * (x*u2x+y*u2y) * (x*v2x+y*v2y) * dx_pml
        A += -R*s/r * u2 * v2 * dx_pml

        B += u0 * v0 * dx + u1 * v1 * dx
        B += u2 * v2 * dx_int

        with ng.TaskManager():
            A.Assemble()
            B.Assemble()

        # Since B is selfadjoint, we do not use SpectralProjNGGeneral here:

        P = SpectralProjNG(X3, A.mat, B.mat,
                           radius=rad, center=ctr, npts=npts,
                           within=within, rhoinv=rhoinv,
                           quadrule=quadrule, inverse=inverse)
        Y = NGvecs(X3, nspan, M=B.mat)
        Yl = Y.create()
        Y.setrandom()
        Yl.setrandom()

        z, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                    **feastkwargs)

        Yg = Y.gridfun()
        Ylg = Y.gridfun()
        y = NGvecs(X, Y.m)
        yl = NGvecs(X, Y.m)
        for i in range(Y.m):
            y._mv[i].data = Yg.components[0].vecs[i]
            yl._mv[i].data = Ylg.components[0].vecs[i]
        y.centernormalize(self.mesh(0, 0))
        yl.centernormalize(self.mesh(0, 0))
        maxbdrnrm = np.max(self.boundarynorm(y))
        print('Mode boundary norm = %.1e' % maxbdrnrm)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return z, yl, y, P, Yl, Y

    # ###################################################################
    # NGSOLVE AUTOMATIC PML #############################################

    def autopmlsystem(self, p, alpha=1):
        """
        Set up PML by NGSolve's automatic PML using in-built
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
                       within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
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
        * Remaining inputs are as documented in leakymode(..).

        OUTPUTS:   zsqr, Yl, Y, P

        * zsqr: computed resonance values Z²
        * Yl, Y: left and right eigenspans
        * P: SpectralProjNGGeneral object that computed Y, Yl
        """

        print('ModeSolver.leakymode called on object with these settings:\n',
              self)

        a, b, X = self.autopmlsystem(p, alpha=alpha)

        P = SpectralProjNGGeneral(X, a.mat, b.mat,
                                  radius=radiusZ2, center=centerZ2, npts=npts,
                                  within=within, rhoinv=rhoinv,
                                  quadrule=quadrule, inverse=inverse)
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
        maxbdrnrm = np.max(self.boundarynorm(Y))
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return zsqr, Yl, Y, beta, P

    # ###################################################################
    # SMOOTHER HANDMADE PML #############################################

    def leakymode_smooth(self, p, radiusZ2=0.1, centerZ2=4,
                         pmlbegin=None, pmlend=None,
                         alpha=1, npts=8, nspan=5, seed=1,
                         within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
                         inverse='umfpack', **feastkwargs):
        """
        Compute leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Then a non-selfadjoint FEAST is run on the system.

        Inputs and outputs are as documented in leakymode_auto(...). The
        only difference is that here you may override the starting and
        ending radius of PML by providing pmlbegin, pmlend.
        """

        print('ModeSolver.leakymode_smooth called on:\n',
              self)
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with poly.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout

        # symbolically derive the radial PML functions
        s, t, R0, R1 = sm.symbols('s t R_0 R_1')
        nr = sm.integrate((s-R0)**2 * (s-R1)**2, (s, R0, t)).factor()
        dr = nr.subs(t, R1).factor()
        sigmat = alpha * nr / dr    # called φ in the docstring
        sigmat = sigmat.subs(R0, pmlbegin).subs(R1, pmlend)
        sigma = sm.diff(t * sigmat, t).factor()
        tau = 1 + 1j * sigma
        taut = 1 + 1j * sigmat
        G = (tau/taut).factor()

        # symbolic -> ngsolve coefficient
        x = ng.x
        y = ng.y
        r = ng.sqrt(x*x+y*y) + 0j
        gstr = str(G).replace('I', '1j').replace('t', 'r')
        ttstr = str(tau*taut).replace('I', '1j').replace('t', 'r')
        g0 = eval(gstr)
        tt0 = eval(ttstr)
        g = ng.IfPos(r-pmlbegin, g0, 1)
        tt = ng.IfPos(r-pmlbegin, tt0, 1)

        gi = 1.0/g
        cs = x/r
        sn = y/r
        A00 = gi*cs*cs+g*sn*sn
        A01 = (gi-g)*cs*sn
        A11 = gi*sn*sn+g*cs*cs
        g.Compile()
        gi.Compile()
        tt.Compile()
        A00.Compile()
        A01.Compile()
        A11.Compile()
        A = ng.CoefficientFunction((A00, A01,
                                    A01, A11), dims=(2, 2))
        self.pml_A = A
        self.pml_tt = tt

        # Make linear eigensystem
        X = ng.H1(self.mesh, order=p, complex=True)
        u, v = X.TnT()
        a = ng.BilinearForm(X)
        b = ng.BilinearForm(X)
        a += (self.pml_A * grad(u) * grad(v) +
              self.V * self.pml_tt * u * v) * dx
        b += self.pml_tt * u * v * dx
        with ng.TaskManager():
            a.Assemble()
            b.Assemble()

        P = SpectralProjNGGeneral(X, a.mat, b.mat,
                                  radius=radiusZ2, center=centerZ2, npts=npts,
                                  within=within, rhoinv=rhoinv,
                                  quadrule=quadrule, inverse=inverse)
        Y = NGvecs(X, 10)
        Yl = NGvecs(X, 10)
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        zsqr, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                       **feastkwargs)
        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))
        maxbdrnrm = np.max(self.boundarynorm(Y))
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return zsqr, Yl, Y, beta, P

    # ###################################################################
    # GUIDED LP MODES FROM SELFADJOINT EIGENPROBLEM #####################

    def selfadjsystem(self, p):

        if self.ngspmlset:
            raise RuntimeError('NGSolve pml mesh trafo set.')
        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()
        A = ng.BilinearForm(X)
        A += grad(u)*grad(v) * dx + self.V*u*v * dx
        B = ng.BilinearForm(X)
        B += u * v * dx
        with ng.TaskManager():
            A.Assemble()
            B.Assemble()

        return A, B, X

    def selfadjmodes(self, interval=(-10, 0), p=3,  seed=1, npts=20, nspan=15,
                     within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
                     verbose=True, inverse='umfpack', **feastkwargs):
        """
        Search for guided modes in a given "interval", which is to be
        input as a tuple: interval=(left, right). These modes solve

        -Δu + V u = Z² u

        with zero dirichlet boundary conditions (no PML, no loss) at the
        outer boundary of the computational domain.

        The computation is done using Lagrangre finite elements of degree "p"
        (with no PML) using selfadjoint FEAST with a random span of "nspan"
        vectors (and using the remaining parameters, which are simply
        passed to feast).

        OUTPUTS:

        betas, Zsqrs, Y:
            betas[i] give the i-th real-valued propagation constant, and
            Zsqrs[i] gives the feast-computed i-th nondimensional Z² value
            in "interval". The corresponding eigenmode is i-th component
            of the span object Y.

        """

        a, b, X = self.selfadjsystem(p)
        left, right = interval
        print('Running selfadjoint FEAST to capture guided modes in ' +
              '({},{})'.format(left, right))
        print('assuming not more than %d modes in this interval' % nspan)
        ctr = (right+left)/2
        rad = (right-left)/2
        P = SpectralProjNG(X, a.mat, b.mat,
                           radius=rad, center=ctr, npts=npts,
                           reduce_sym=True, within=within, rhoinv=rhoinv,
                           quadrule=quadrule, inverse=inverse, verbose=verbose)
        Y = NGvecs(X, nspan, M=b.mat)
        Y.setrandom(seed=seed)
        Zsqrs, Y, history, _ = P.feast(Y, hermitian=True, **feastkwargs)
        betas = self.betafrom(Zsqrs)

        return betas, Zsqrs, Y

    # ###################################################################
    # VECTOR MODES

    def vecmodesystem(self, p, alpha=None, d=None):
        """
        Prepare eigensystem and resolvents for solving for vector modes.

        INPUTS:

        p: Determines degree of Nedelec x Lagrange space system.
           This should be an integer >= 0.

        alpha: If alpha is None, prepare system for vector guided modes.
           If alpha is a positive number, use it as PML strength and
           prepare system for leaky modes using NGSolve's automatic
           mesh-based PML.
        """

        if alpha is not None:
            self.ngspmlset = True
            if d is not None:
                self.set_smooth_ngpml(alpha, d)
            else:
                radial = ng.pml.Radial(rad=self.R,
                                       alpha=alpha*1j, origin=(0, 0))
                self.mesh.SetPML(radial, 'Outer')
                print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                      'and thickness=%.3f' % (self.Rout-self.R))
        elif self.ngspmlset:
            raise RuntimeError('Unexpected NGSolve pml mesh trafo here.')

        n = self.index
        n2 = n*n
        X = ng.HCurl(self.mesh, order=p+1-max(1-p, 0), type1=True,
                     dirichlet='OuterCircle', complex=True)
        Y = ng.H1(self.mesh, order=p+1, dirichlet='OuterCircle', complex=True)
        E, v = X.TnT()
        phi, psi = Y.TnT()

        A = ng.BilinearForm(X)
        A += (curl(E) * curl(v) + self.V * E * v) * dx
        M = ng.BilinearForm(X)
        M += E * v * dx
        C = ng.BilinearForm(trialspace=Y, testspace=X)
        C += grad(phi) * v * dx
        B = ng.BilinearForm(trialspace=X, testspace=Y)
        B += -n2 * E * grad(psi) * dx
        D = ng.BilinearForm(Y)
        D += n2 * phi * psi * dx

        with ng.TaskManager():
            try:
                A.Assemble()
                M.Assemble()
                B.Assemble()
                C.Assemble()
                D.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                M.Assemble()
                B.Assemble()
                C.Assemble()
                D.Assemble()
            Dinv = D.mat.Inverse(Y.FreeDofs())

        # resolvent of the vector mode problem --------------------------

        class ResolventVectorMode():

            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY)
            wrk2 = ng.GridFunction(XY)
            tmpY1 = ng.GridFunction(Y)
            tmpY2 = ng.GridFunction(Y)
            tmpX1 = ng.GridFunction(X)

            def __init__(selfr, z, V, n):
                n2 = n*n
                XY = ng.FESpace([X, Y])
                (E, phi), (v, psi) = XY.TnT()
                selfr.zminusOp = ng.BilinearForm(XY)
                selfr.zminusOp += (z * E * v - curl(E) * curl(v)
                                   - V * E * v - grad(phi) * v
                                   - n2 * phi * psi + n2 * E * grad(psi)) * dx
                with ng.TaskManager():
                    try:
                        selfr.zminusOp.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.zminusOp.Assemble()
                    selfr.R = selfr.zminusOp.mat.Inverse(XY.FreeDofs())

            def act(selfr, v, Rv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]

                with ng.TaskManager():
                    Mv[:] = M.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0
                        selfr.wrk2.vec.data = selfr.R * selfr.wrk1.vec
                        Rv._mv[i][:] = selfr.wrk2.components[0].vec
                    # Rv.zerobdry()

            def adj(selfr, v, RHv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]
                with ng.TaskManager():
                    Mv[:] = M.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0
                        selfr.wrk2.vec.data = selfr.R.H * selfr.wrk1.vec
                        RHv._mv[i][:] = selfr.wrk2.components[0].vec
                    # RHv.zerobdry()

            def rayleigh_nsa(selfr, ql, qr, qAq=not None, qBq=not None,
                             workspace=None):
                """
                Return qAq[i, j] = (𝒜 qr[j], ql[i]) with 𝒜 =  (A - C D⁻¹ B) E
                and qBq[i, j] = (M qr[j], ql[i]). """

                if workspace is None:
                    Aqr = ng.MultiVector(qr._mv[0], qr.m)
                else:
                    Aqr = workspace._mv[:qr.m]

                with ng.TaskManager():
                    if qAq is not None:
                        Aqr[:] = A.mat * qr._mv
                        for i in range(qr.m):
                            selfr.tmpY1.vec.data = B.mat * qr._mv[i]
                            selfr.tmpY2.vec.data = Dinv * selfr.tmpY1.vec
                            selfr.tmpX1.vec.data = C.mat * selfr.tmpY2.vec
                            Aqr[i].data -= selfr.tmpX1.vec
                        qAq = ng.InnerProduct(Aqr, ql._mv).NumPy().T

                    if qBq is not None:
                        Bqr = Aqr
                        Bqr[:] = M.mat * qr._mv
                        qBq = ng.InnerProduct(Bqr, ql._mv).NumPy().T

                return (qAq, qBq)

            def rayleigh(selfr, q, workspace=None):
                return selfr.rayleigh_nsa(q, q, workspace=workspace)

        # resolvent class definition done -------------------------------

        return ResolventVectorMode, M.mat, A.mat, B.mat, C.mat, D.mat, Dinv

    def guidedvecmodes(self, rad, ctr, p=3,  seed=None, npts=8, nspan=20,
                       within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
                       verbose=True, inverse='umfpack',
                       **feastkwargs):
        """
        Capture guided vector modes whose non-dimensional resonance value Z²
        is such that Z*Z is within the interval (ctr-rad, ctr+rad).
        """

        R, M, A, B, C, D, Dinv = self.vecmodesystem(p)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        E.setrandom(seed=seed)

        print('Using FEAST to search for vector guided modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval.' % nspan)
        print('System size:', E.n, ' x ', E.n, '  Inverse type:', inverse)

        P = SpectralProjNGR(lambda z: R(z, self.V, self.index),
                            radius=rad, center=ctr, npts=npts,
                            within=within, rhoinv=rhoinv, quadrule=quadrule,
                            inverse=inverse, verbose=verbose)
        Zsqrs, E, history, _ = P.feast(E, **feastkwargs)
        betas = self.betafrom(Zsqrs)

        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv
        phi._mv[:] = Dinv * BE._mv

        return betas, Zsqrs, E, phi, R

    def leakyvecmodes(self, rad, ctr, alpha=1, p=3,  seed=1, npts=8, nspan=20,
                      within=None, rhoinv=0.0, quadrule='circ_trapez_shift',
                      verbose=True, inverse='umfpack', d=None,
                      **feastkwargs):
        """
        Capture leaky vector modes whose non-dimensional resonance value Z²
        is contained  within the circular contour centered at "ctr"
        of radius "rad" in the Z² complex plane (not the Z-plane!).
        """

        R, M, A, B, C, D, Dinv = self.vecmodesystem(p, alpha=alpha, d=d)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        El = E.create()
        E.setrandom(seed=seed)
        El.setrandom(seed=seed)

        print('Using FEAST to search for vector leaky modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval' % nspan)
        print('System size:', E.n, ' x ', E.n, '  Inverse type:', inverse)

        P = SpectralProjNGR(lambda z: R(z, self.V, self.index),
                            radius=rad, center=ctr, npts=npts,
                            within=within, rhoinv=rhoinv, quadrule=quadrule,
                            inverse=inverse, verbose=verbose)

        Zsqrs, E, history, El = P.feast(E, Yl=El, hermitian=False,
                                        **feastkwargs)
        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv
        phi._mv[:] = Dinv * BE._mv

        betas = self.betafrom(Zsqrs)
        print('Results:\n Z²:', Zsqrs)
        print(' beta:', betas)
        print(' CL dB/m:', 20 * betas.imag / np.log(10))

        return betas, Zsqrs, E, phi, R

    def set_smooth_ngpml(self, alpha, d):
        """NGSolve custom pml of form:

            r = rh + 1j * α * φ(rh - R))

        where

            φ(rh - R)) = 1/(d+1) * alpha * 1j ((rh - R) / W)^(d+1).

        Integer d determines the degee of smoothing.
        """
        if d < 0:
            raise ValueError('PML degree must be integer >= 0.')

        W = self.Rout-self.R

        # Set complex trafo
        rh = ng.sqrt(ng.x**2 + ng.y**2)
        r = rh + 1/(d+1) * alpha * 1j * ((rh - self.R) / W)**(d+1)
        drdrh = 1 + alpha*1j / d * ((rh - self.R) / W)**d

        x = r * ng.x / rh
        y = r * ng.y / rh

        trafo = ng.CoefficientFunction((x, y))

        # derivatives and determinant

        dxdxh = drdrh * (ng.x / rh) ** 2 + r * (ng.y ** 2 / rh ** 3)
        dxdyh = drdrh * (ng.y * ng. x / rh ** 2) - \
            r * (ng.y * ng.x / rh ** 3)

        dydyh = drdrh * (ng.y / rh) ** 2 + r * (ng.x ** 2 / rh ** 3)
        dydxh = dxdyh

        J = ng.CoefficientFunction((dxdxh, dxdyh, dydxh, dydyh),
                                   dims=(2, 2))

        custom_pml = ng.pml.Custom(trafo, J)

        self.mesh.SetPML(custom_pml, 'Outer')
        print('Set NGSolve custom (smooth) PML with alpha=', alpha, 'degree=',
              d, 'and thickness=%.3f' % (self.Rout-self.R))
