"""
Definition of ModeSolver class and its methods for computing
modes of various fibers.
"""
from warnings import warn
from ngsolve import curl, grad, dx, Conj, Integrate, InnerProduct, CF, Grad
from numpy import conj
from pyeigfeast import NGvecs, SpectralProjNG
from pyeigfeast import SpectralProjNGR, SpectralProjNGPoly
from fiberamp.fiber.utilities import Strategy, AdaptivityStrategy
import ngsolve as ng
import numpy as np
import sympy as sm


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
        # Set gamma to None, so that it can be set later *once*
        # These coefficients will be members, and will be set by
        # set_vecpml_coeff.  Gabriel
        self.gamma = None

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
        return np.sqrt((self.L * self.k * self.n0)**2 - Z2) / self.L

    def sqrZfrom(self, betas):
        """ Return values of nondimensional Z squared, given physical
        propagation constants betas, ie, return Z² = L² (k²n₀² - β²). """
        return (self.L * self.k * self.n0)**2 - (self.L * betas)**2

    def boundarynorm(self, y):
        """
        Returns  L² norm of all functions in the span y restricted to
        the outermost boundary r= Rout.
        """

        def outint(u):
            dl = dx(definedon=self.mesh.Boundaries('OuterCircle'))
            s = abs(ng.Integrate(u * ng.Conj(u) * dl, self.mesh))
            return np.sqrt(s)

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
        bdryval = np.exp(-decayrate) / np.sqrt(np.abs(Z) * np.pi / 2)
        bdrnrm0 = bdryval * 2 * np.pi * self.Rout
        print('PML decay estimates boundary norm ~ %.1e' % max(bdrnrm0))
        return bdrnrm0

    def power(self, Etv, phi, beta):
        """
        Find power of mode with transverse electric and magnetic fields E, H.
        If E and H are from different modes, this method finds 'inner product'
        of the two modes according to the orthogonality type relationship
        found in Marcuse, Light Transmission Optics 2nd edition, eq 8.5.12 (and
        also in Snyder's Optical Waveguide Theory equation 11-13).

        Parameters
        ----------
        E : ngsolve coefficient function
            Transverse electric field E = (Ex, Ey) (or (Er, Ey) for bent modes)
        H : same type as E
            Magnetic field of second mode (organized in same way).

        Returns
        -------
        p : float or complex
            Power through transverse plane (at z = 0).

        """
        Sz = self.S(Etv, phi, beta)[1]
        p = ng.Integrate(Sz, self.mesh)
        return p

    def maxwell_product(self, E1, H2):
        """
        Find maxwell type product of two modes with transverse electric and
        magnetic fields E1, H2 respectively.

        This method finds 'inner product' of the two modes according to the
        orthogonality type relationship found in Marcuse, Light Transmission
        Optics 2nd edition, eq 8.5.12 (and also in Snyder's Optical Waveguide
        Theory equation 11-13).

        Parameters
        ----------
        E : ngsolve coefficient function
            Transverse electric field E = (Ex, Ey) (or (Er, Ey) for bent modes)
        H : same type as E
            Magnetic field of second mode (organized in same way).

        Returns
        -------
        p : float or complex
            Returns power if E and H belong to same mode.  Returns 'dot
            product type' value if E and H belong to distinct modes.

        """
        Sz = E1[0] * Conj(H2[1]) - E1[1] * Conj(H2[0])
        p = 1 / 2 * ng.Integrate(Sz, self.mesh)
        return p

    def H_field_from_E(self, Etv, phi, beta):
        """Return the H field (Htv, Hz) from transverse E field and
        phi = i beta Ez (found using vector modesolver).

        We assume the beta is provided unscaled, and we then non-dimensionalize
        to get beta_s = beta * self.L and k_s = self.k * self.L.  This is then
        used to form the H field.  These give the appropriate values for the
        non-dimensionalized mesh we are using.

        In order to avoid numeric difficulties, we scale the H field by the
        negative imaginary unit times the vacuum impedence 𝜂0 defined by
        𝜂0 := (𝜇0/𝜀0)^(1/2).

        This transforms Maxwell's equations to give

                    curl E = k0 H
                    curl H = k0 e_r E

        where e_r is the relative permittivity.
        """
        beta_s = beta * self.L
        k_s = self.k * self.L
        dphi_dx, dphi_dy = grad(phi)

        J_Etv = ng.CF((Etv[1], -Etv[0]))
        rot_phi = ng.CF((dphi_dy, -dphi_dx))

        Htv = -1j / (k_s * beta_s) * (rot_phi + beta_s**2 * J_Etv)
        Hz = 1 / k_s * curl(Etv)

        return Htv, Hz

    def S(self, Etv, phi, beta):
        """Return time averaged Poynting vector S = 1/2 E x H*.

        Here we again scale the H field by -1j * 𝜂0 with 𝜂0 defined by
        𝜂0 := (𝜇0/𝜀0)^(1/2).

        This transforms Maxwell's equations to give

                    curl E = k0 H
                    curl H = k0 e_r E

        where e_r is the relative permittivity.
        """
        beta_s = beta * self.L
        k_s = self.k * self.L

        J_Etv = ng.CF((Etv[1], -Etv[0]))

        # Stv = J_Etv * Conj(curl(Etv)) + phi / (k_s *
        # beta_s * conj(beta_s)) * \
        #     (Conj(grad(phi)) + conj(beta_s)**2 * Conj(Etv))

        Stv = -1j * (J_Etv * Conj(curl(Etv)) + np.abs(beta_s)**-2 * phi *
                     Conj(grad(phi)) + conj(beta_s) / beta_s * phi * Conj(Etv))

        Sz = 1 / (k_s * conj(beta_s)) * \
            (Etv * Conj(grad(phi)) + conj(beta_s)**2 * Etv.Norm()**2)

        return 1 / 2 * Stv, 1 / 2 * Sz

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
        r = ng.sqrt(x * x + y * y) + 0j
        X = ng.H1(self.mesh, order=p, complex=True)
        u, v = X.TnT()
        ux, uy = grad(u)
        vx, vy = grad(v)

        AA = [ng.BilinearForm(X, check_unused=False)]
        AA[0] += (s * r / R) * grad(u) * grad(v) * dx_pml
        AA[0] += s * (r - R) / (R * r * r) * (x * ux + y * uy) * v * dx_pml
        AA[0] += s * (R - 2 * r) / r**3 * (x * ux + y * uy) * (x * vx +
                                                               y * vy) * dx_pml
        AA[0] += -s**3 * (r - R)**2 / (R * r) * u * v * dx_pml

        AA += [ng.BilinearForm(X)]
        AA[1] += grad(u) * grad(v) * dx_int
        AA[1] += self.V * u * v * dx_int
        AA[1] += 2 * (r - R) / r**3 * (x * ux + y * uy) * (x * vx +
                                                           y * vy) * dx_pml
        AA[1] += 1 / r**2 * (x * ux + y * uy) * v * dx_pml
        AA[1] += -2 * s * s * (r - R) / r * u * v * dx_pml

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[2] += R / s / r**3 * (x * ux + y * uy) * (x * vx + y * vy) * dx_pml
        AA[2] += -R * s / r * u * v * dx_pml

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

    def leakymode(self,
                  p,
                  ctr=2,
                  rad=0.1,
                  alpha=1,
                  npts=8,
                  within=None,
                  rhoinv=0.0,
                  quadrule='circ_trapez_shift',
                  nspan=5,
                  seed=1,
                  inverse=None,
                  verbose=True,
                  **feastkwargs):
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
              'and thickness=%.3f' % (self.Rout - self.R))

        Y = NGvecs(X3, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)

        P = SpectralProjNGPoly(AA,
                               X,
                               radius=rad,
                               center=ctr,
                               npts=npts,
                               within=within,
                               rhoinv=rhoinv,
                               quadrule=quadrule,
                               verbose=verbose,
                               inverse=inverse)

        Z, Y, hist, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)
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

        moreoutputs = {
            'longY': Y,
            'longYl': Yl,
            'ewshistory': ews,
            'bdrnorm': bdrnrm,
            'converged': cgd
        }

        return Z, y, yl, beta, P, moreoutputs

    def leakymode_poly(self,
                       p,
                       ctr=2,
                       rad=0.1,
                       alpha=1,
                       npts=8,
                       within=None,
                       rhoinv=0.0,
                       quadrule='circ_trapez_shift',
                       nspan=5,
                       seed=1,
                       inverse=None,
                       **feastkwargs):
        """
        This method is an alternate implementation of the polynomial
        eigensolver using NGSolve bilinear forms in a product finite
        element space. It has been useful sometimes in testing and
        debugging. It should give the same results as leakymode(...),
        and its arguments are as documented in leakymode(...).
        It's more expensive than leakymode(...).
        """

        print('ModeSolver.leakymode_poly called on this object:\n', self)
        print('Set freq-dependent PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout - self.R))
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
        r = ng.sqrt(x * x + y * y) + 0j

        A = ng.BilinearForm(X3)
        B = ng.BilinearForm(X3)

        A += u1 * v0 * dx
        A += u2 * v1 * dx

        A += (s * r / R) * grad(u0) * grad(v2) * dx_pml
        A += s * (r - R) / (R * r * r) * (x * u0x + y * u0y) * v2 * dx_pml
        A += s * (R - 2 * r) / r**3 * (x * u0x + y * u0y) * (x * v2x +
                                                             y * v2y) * dx_pml
        A += -s**3 * (r - R)**2 / (R * r) * u0 * v2 * dx_pml

        A += grad(u1) * grad(v2) * dx_int
        A += self.V * u1 * v2 * dx_int
        A += 2 * (r - R) / r**3 * (x * u1x + y * u1y) * (x * v2x +
                                                         y * v2y) * dx_pml
        A += 1 / r**2 * (x * u1x + y * u1y) * v2 * dx_pml
        A += -2 * s * s * (r - R) / r * u1 * v2 * dx_pml

        A += R / s / r**3 * (x * u2x + y * u2y) * (x * v2x + y * v2y) * dx_pml
        A += -R * s / r * u2 * v2 * dx_pml

        B += u0 * v0 * dx + u1 * v1 * dx
        B += u2 * v2 * dx_int

        with ng.TaskManager():
            try:
                A.Assemble()
                B.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                B.Assemble()

        P = SpectralProjNG(X3,
                           A.mat,
                           B.mat,
                           radius=rad,
                           center=ctr,
                           npts=npts,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           inverse=inverse)
        Y = NGvecs(X3, nspan, M=B.mat)
        Yl = Y.create()
        Y.setrandom()
        Yl.setrandom()

        z, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)

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

        radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
        self.mesh.SetPML(radial, 'Outer')
        print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout - self.R))
        X = ng.H1(self.mesh, order=p, complex=True)

        u, v = X.TnT()
        a = ng.BilinearForm(X)
        b = ng.BilinearForm(X)
        a += (grad(u) * grad(v) + self.V * u * v) * dx
        b += u * v * dx
        with ng.TaskManager():
            try:
                a.Assemble()
                b.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                a.Assemble()
                b.Assemble()

        return a, b, X

    def leakymode_auto(self,
                       p,
                       radiusZ2=0.1,
                       centerZ2=4,
                       alpha=1,
                       npts=8,
                       nspan=5,
                       seed=1,
                       within=None,
                       rhoinv=0.0,
                       verbose=True,
                       quadrule='circ_trapez_shift',
                       inverse='umfpack',
                       **feastkwargs):
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

        OUTPUTS:   zsqr, Yr, Yl, P

        * zsqr: computed resonance values Z²
        * Yl, Yr: left and right eigenspans
        * P: spectral projector object that computed Y, Yl
        """

        print('ModeSolver.leakymode called on object with these settings:\n',
              self)

        a, b, X = self.autopmlsystem(p, alpha=alpha)

        P = SpectralProjNG(X,
                           a.mat,
                           b.mat,
                           radius=radiusZ2,
                           center=centerZ2,
                           checks=False,
                           npts=npts,
                           within=within,
                           rhoinv=rhoinv,
                           verbose=verbose,
                           quadrule=quadrule,
                           inverse=inverse)

        Y = NGvecs(X, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        zsqr, Y, _, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)
        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))
        maxbdrnrm = np.max(self.boundarynorm(Y))
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return zsqr, Y, Yl, beta, P

    # ###################################################################
    # SMOOTHER HANDMADE PML #############################################

    # ###################################################################
    # # SYMBOLICS AND RESOLVENT #########################################

    def smoothpmlsymb(self, alpha, pmlbegin, pmlend):
        """
        Symbolic pml functions useful for debugging/visualization of pml.
        ---
        We compute a radial PML function φ(r) = α * φ(r) and the derived
        functions τ(r) = μ(r) = 1 + αφ(r) (called taut in the code),
        and τ_mapped(r) = r * τ(r) = r * μ(r) = η(r) = r * (1 + αφ(r)).
        Remaining terms are also computed. Notation is not unified.
        Reader is advised to consult the code for the exact meaning of
        the symbols used.
        Cf. Kim and Pasciak; Gopalakrishnan et al.
        """
        # symbolically derive the radial PML functions
        s, t, r0, r1 = sm.symbols('s t R_0 R_1')
        nr = sm.integrate((s - r0)**2 * (s - r1)**2, (s, r0, t)).factor()
        dr = nr.subs(t, r1).factor()
        phi = alpha * nr / dr  # called α * φ in the docstring
        phi = phi.subs(r0, pmlbegin).subs(r1, pmlend)
        # Remaining terms
        sigma = sm.diff(t * phi, t).factor()
        tau = 1 + 1j * sigma
        taut = 1 + 1j * phi  # called μ in the docstring
        mappedt = t * taut  # called η in the docstring
        g = (tau / taut).factor()  # this is what appears in the mapped system
        return g, mappedt, tau, taut

    def symb_to_cf(self, symb, r=None):
        """
        Convert a symbolic expression to an ngsolve coefficient function.
        If r is None, then the symbolic expression is assumed to be the radius.
        Otherwise, r is assumed to be a valid ngsolve coefficient function.
        Assumes that the symbolic expression is a function of t, and that
        the imaginary unit is I.
        """
        x = ng.x
        y = ng.y
        if r is None:
            r = ng.sqrt(x * x + y * y)
        # TODO How to assert r is a valid ngsolve coefficient function?
        strng = str(symb).replace('I', '1j').replace('t', 'r')
        cf = eval(strng)
        return cf

    def set_vecpml_coeff(self, alpha, pmlbegin, pmlend, **kwargs):
        """
        Set the PML coefficients. Function defined to reduce redundancy
        and improve readability.
        Adds the following attributes to the class:
        * self.detj
        * self.detj_conj
        * self.kappa
        * self.kappa_conj
        * self.gamma
        * self.gamma_conj
        Check documentation of CF.Compile for kwargs.
        Recommended realcompile=True and wait=True.
        """
        # Standard ngsolve imports
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)
        # Get symbolic functions
        _, eta_sym, _, mu_sym = self.smoothpmlsymb(alpha, pmlbegin, pmlend)
        t = sm.symbols('t')
        eta_dt = sm.diff(eta_sym, t).factor()
        mu_dt = sm.diff(mu_sym, t).factor()
        # mu_dt = sm.diff(t * mu_sym, t).factor()
        # Make coefficient functions
        mu_ = self.symb_to_cf(mu_sym)
        eta_ = self.symb_to_cf(eta_sym)
        mu_dr_ = self.symb_to_cf(mu_dt)
        eta_dr_ = self.symb_to_cf(eta_dt)
        # Main terms, after truncating at pmlbegin
        mu = ng.IfPos(r - pmlbegin, mu_, 1)
        eta = ng.IfPos(r - pmlbegin, eta_, r)
        mu_dr = ng.IfPos(r - pmlbegin, mu_dr_, 0)
        eta_dr = ng.IfPos(r - pmlbegin, eta_dr_, 1)
        # Determinant of Jacobian
        detj = mu * eta_dr
        # Jacobian, left as a reminder
        # # j00 = mu + (mu_dr / r) * x * x
        # # j01 = - (mu_dr / r) * x * y
        # # j11 = mu + (mu_dr / r) * y * y
        # Inverse of Jacobian
        jinv00 = 1 / eta_dr + (mu_dr / (eta_dr * eta)) * y * y
        jinv01 = -(mu_dr / (eta_dr * eta)) * x * y
        jinv11 = 1 / eta_dr + (mu_dr / (eta_dr * eta)) * x * x
        # Conjugate the main terms
        mu_conj = ng.Conj(mu)
        eta_conj = ng.Conj(eta)
        mu_dr_conj = ng.Conj(mu_dr)
        eta_dr_conj = ng.Conj(eta_dr)
        detj_conj = ng.Conj(detj)
        jinv00_conj = ng.Conj(jinv00)
        jinv01_conj = ng.Conj(jinv01)
        jinv11_conj = ng.Conj(jinv11)
        # Compile into coefficient functions
        # Only compile the main terms
        jinv00.Compile(**kwargs)
        jinv01.Compile(**kwargs)
        jinv11.Compile(**kwargs)
        detj.Compile(**kwargs)
        detj_conj.Compile(**kwargs)
        jinv00_conj.Compile(**kwargs)
        jinv01_conj.Compile(**kwargs)
        jinv11_conj.Compile(**kwargs)
        # Construct jacobians
        # jac = ng.CoefficientFunction((j00, j01, j01, j11), dims=(2, 2))
        jacinv = ng.CoefficientFunction((jinv00, jinv01, jinv01, jinv11),
                                        dims=(2, 2))
        jacinv_conj = ng.CoefficientFunction(
            (jinv00_conj, jinv01_conj, jinv01_conj, jinv11_conj), dims=(2, 2))
        # Construct gamma, gamma_conj, kappa, kappa_conj
        gamma = detj * (jacinv * jacinv)
        gamma_conj = detj_conj * (jacinv_conj * jacinv_conj)
        kappa = (mu / eta_dr**3) * (1 + (mu_dr * r**2) / eta)**2
        kappa_conj = (mu_conj / eta_dr_conj**3) * \
            (1 + (mu_dr_conj * r**2) / eta_conj)**2

        # Adding  terms to the class as needed
        if self.gamma is not None:
            raise RuntimeError('PML coefficients already set.'
                               ' Check code logic.')
        gamma.Compile(**kwargs)
        gamma_conj.Compile(**kwargs)
        kappa.Compile(**kwargs)
        kappa_conj.Compile(**kwargs)
        # Set the coefficients
        setattr(self, 'detj', detj)
        setattr(self, 'detj_conj', detj_conj)
        setattr(self, 'kappa', kappa)
        setattr(self, 'kappa_conj', kappa_conj)
        setattr(self, 'gamma', gamma)
        setattr(self, 'gamma_conj', gamma_conj)

    def make_resolvent_maxwell(self,
                               m,
                               a,
                               b,
                               c,
                               d,
                               X,
                               Y,
                               inverse='umfpack',
                               autoupdate=True):
        """
        Create resolvent for Maxwell's equations.
        INPUTS:
        * a, b, c, d: bilinear forms of the system, blockwise form
        * X, Y: FE spaces
        * inverse: inverse type
        * autoupdate: ngsolve autoupdate
        OUTPUTS:
        * ResolventVectorMode: resolvent
        * dinv: inverse of d
        """
        print('ModeSolver.make_resolvent_maxwell called...\n')
        # Retrive coefficient functions
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)

        mu = self.mu
        mu_dr = self.mu_dr
        eta = self.eta
        eta_dr = self.eta_dr
        detj = self.detj
        jacinv = self.jacinv

        # Create inverse of d
        dinv = d.mat.Inverse(Y.FreeDofs(coupling=True), inverse=inverse)

        # resolvent class definition begins here ------------------------------
        class ResolventVectorMode():
            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY,
                                   name='wrk1',
                                   autoupdate=autoupdate,
                                   nested=autoupdate)
            wrk2 = ng.GridFunction(XY,
                                   name='wrk2',
                                   autoupdate=autoupdate,
                                   nested=autoupdate)
            tmpY1 = ng.GridFunction(Y,
                                    name='tmpY1',
                                    autoupdate=autoupdate,
                                    nested=autoupdate)
            tmpY2 = ng.GridFunction(Y,
                                    name='tmpY2',
                                    autoupdate=autoupdate,
                                    nested=autoupdate)
            tmpX1 = ng.GridFunction(X,
                                    name='tmpX1',
                                    autoupdate=autoupdate,
                                    nested=autoupdate)

            def __init__(selfr, z, V, n, inverse=None):
                n2 = n * n
                XY = ng.FESpace([X, Y])

                (E, phi), (F, psi) = XY.TnT()

                selfr.Z = ng.BilinearForm(XY, condense=True)
                selfr.ZH = ng.BilinearForm(XY, condense=True)

                # m - a - c - b + (-d)
                selfr.Z += (z * detj * (jacinv * E) * (jacinv * F) -
                            (mu / eta_dr**3) *
                            (1 + (mu_dr * r**2) / eta)**2 * curl(E) * curl(F) -
                            V * detj * (jacinv * E) * (jacinv * F) - detj *
                            (jacinv * grad(phi)) * (jacinv * F) - n2 * detj *
                            (jacinv * E) *
                            (jacinv * grad(psi)) + n2 * detj * phi * psi) * dx
                selfr.ZH += (np.conjugate(z) * detj * (jacinv * F) *
                             (jacinv * E) - (mu / eta_dr**3) *
                             (1 +
                              (mu_dr * r**2) / eta)**2 * curl(F) * curl(E) -
                             V * detj * (jacinv * F) *
                             (jacinv * E) - n2 * detj * (jacinv * F) *
                             (jacinv * grad(phi)) - detj *
                             (jacinv * grad(psi)) *
                             (jacinv * E) + n2 * detj * phi * psi) * dx

                with ng.TaskManager():
                    try:
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    selfr.R_I = selfr.Z.mat.Inverse(
                        selfr.XY.FreeDofs(coupling=True), inverse=inverse)

            def act(selfr, v, Rv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]

                with ng.TaskManager():
                    Mv[:] = m.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0

                        # selfr.wrk2.vec.data = selfr.R * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.Z.harmonic_extension_trans * \
                            selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.harmonic_extension * selfr.wrk2.vec

                        Rv._mv[i][:] = selfr.wrk2.components[0].vec

            def adj(selfr, v, RHv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]
                with ng.TaskManager():
                    Mv[:] = m.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0

                        # selfr.wrk2.vec.data = selfr.R.H * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.ZH.harmonic_extension_trans * \
                            selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I.H * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.harmonic_extension * selfr.wrk2.vec

                        RHv._mv[i][:] = selfr.wrk2.components[0].vec

            def rayleigh_nsa(selfr,
                             ql,
                             qr,
                             qAq=not None,
                             qBq=not None,
                             workspace=None):
                """
                Return qAq[i, j] = (𝒜 qr[j], ql[i]) with
                𝒜 =  (A - C D⁻¹ B) E
                and qBq[i, j] = (M qr[j], ql[i]).
                """
                if workspace is None:
                    Aqr = ng.MultiVector(qr._mv[0], qr.m)
                else:
                    Aqr = workspace._mv[:qr.m]

                with ng.TaskManager():
                    if qAq is not None:
                        Aqr[:] = a.mat * qr._mv
                        for i in range(qr.m):
                            # TODO: Static condensation caused issues when
                            #       not using TaskManager
                            selfr.tmpY1.vec.data = b.mat * qr._mv[i]
                            selfr.tmpY1.vec.data += \
                                d.harmonic_extension_trans * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data = dinv * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                d.inner_solve * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                d.harmonic_extension * selfr.tmpY2.vec

                            selfr.tmpX1.vec.data = c.mat * selfr.tmpY2.vec
                            Aqr[i].data -= selfr.tmpX1.vec
                        qAq = ng.InnerProduct(Aqr, ql._mv).NumPy().T

                    if qBq is not None:
                        Bqr = Aqr
                        Bqr[:] = m.mat * qr._mv
                        qBq = ng.InnerProduct(Bqr, ql._mv).NumPy().T

                return (qAq, qBq)

            def rayleigh(selfr, q, workspace=None):
                return selfr.rayleigh_nsa(q, q, workspace=workspace)

            def update_system(selfr, verbose=False):
                """
                Update the system matrices.
                For internal use only.
                This should be redundant with the autoupdate feature of
                the GridFunction objects and recreating the resolvent
                object should not be necessary.
                Consider removing this method and simplyfing __init__.
                """
                warn(
                    'This should be redundant with the autoupdate feature of'
                    ' the GridFunction objects and recreating the resolvent'
                    ' object should not be necessary.',
                    PendingDeprecationWarning)
                if verbose:
                    print('Updating system matrices...\n')
                with ng.TaskManager():
                    try:
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    except Exception:
                        if verbose:
                            print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    selfr.R_I = selfr.Z.mat.Inverse(
                        selfr.XY.FreeDofs(coupling=True),
                        inverse=selfr.inverse)

        # end of class ResolventVectorMode --------------------------------

        return ResolventVectorMode, dinv

    # ###################################################################
    # # PML SYSTEMS #####################################################

    def smoothpmlsystem(self,
                        p,
                        alpha=1,
                        pmlbegin=None,
                        pmlend=None,
                        autoupdate=False):
        """
        Make the matrices needed for formulating the leaky mode
        eigensystem with frequency-independent C² PML map
            mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r.
        """

        print('ModeSolver.leakymode_smooth called on:\n', self)
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with smooth.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout

        G, mappedt, tau, taut = self.smoothpmlsymb(alpha, pmlbegin, pmlend)

        # symbolic -> ngsolve coefficient
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)
        gstr = str(G).replace('I', '1j').replace('t', 'r')
        ttstr = str(tau * taut).replace('I', '1j').replace('t', 'r')
        self.ttstr = ttstr
        self.taut = str(taut).replace('I', '1j').replace('t', 'r')
        self.mappedr = str(mappedt).replace('I', '1j').replace('t', 'r')
        g0 = eval(gstr)
        tt0 = eval(ttstr)
        g = ng.IfPos(r - pmlbegin, g0, 1)
        tt = ng.IfPos(r - pmlbegin, tt0, 1)

        gi = 1.0 / g
        cs = x / r
        sn = y / r
        A00 = gi * cs * cs + g * sn * sn
        A01 = (gi - g) * cs * sn
        A11 = gi * sn * sn + g * cs * cs
        g.Compile()
        gi.Compile()
        tt.Compile()
        A00.Compile()
        A01.Compile()
        A11.Compile()
        A = ng.CoefficientFunction((A00, A01, A01, A11), dims=(2, 2))
        self.pml_A = A
        self.pml_B = tt

        # Make linear eigensystem
        X = ng.H1(self.mesh, order=p, complex=True, autoupdate=autoupdate)
        u, v = X.TnT()
        a = ng.BilinearForm(X)
        b = ng.BilinearForm(X)
        a += (self.pml_A * grad(u) * grad(v) +
              self.V * self.pml_B * u * v) * dx
        b += self.pml_B * u * v * dx

        with ng.TaskManager():
            try:
                a.Assemble()
                b.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                a.Assemble()
                b.Assemble()

        return a, b, X

    def smoothvecpmlsystem_compound(self,
                                    p,
                                    alpha=1,
                                    pmlbegin=None,
                                    pmlend=None,
                                    deg=2,
                                    autoupdate=True):
        """
        Make the matrices needed for formulating the vector
        leaky mode eigensystem with frequency-independent
        C² PML map mapped_x = x * (1 + 1j * α * φ(r)) where
        φ is a C² function of the radius r.
        Using the compound finite element space X*Y.
        INPUTS:
        * p: polynomial degree of finite elements
        * alpha: PML strength
        * pmlbegin: radius where PML begins
        * pmlend: radius where PML ends
        * deg: degree of the PML polynomial
        * autoupdate: whether to use autoupdate in NGSolve
        OUTPUTS:
        * aa: bilinear form for the LHS
        * mm: bilinear form for the RHS
        * Z: finite element space
        """
        if self.ngspmlset:
            raise RuntimeError(
                'NGSolve PML set. Cannot combine with smooth PML.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout
        if self.gamma is None:
            self.set_vecpml_coeff(alpha, pmlbegin, pmlend, maxderiv=3)
        self.p = p

        # Get symbolic functions
        detj = self.detj
        kappa = self.kappa
        gamma = self.gamma

        # Make linear eigensystem, cf. self.vecmodesystem
        n2 = self.index * self.index
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True,
                     autoupdate=autoupdate)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True,
                  autoupdate=autoupdate)

        Z = X * Y
        (E, phi), (F, psi) = Z.TnT()

        aa = ng.BilinearForm(Z)
        mm = ng.BilinearForm(Z)

        aa += ((kappa * curl(E)) * curl(F) + self.V * (gamma * E) * F + n2 *
               (gamma * E) * grad(psi) +
               (gamma * grad(phi)) * F - n2 * detj * phi * psi) * dx
        mm += (gamma * E) * F * dx

        with ng.TaskManager():
            try:
                aa.Assemble()
                mm.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                aa.Assemble()
                mm.Assemble()

        return aa, mm, Z

    def smoothvecpmlsystem_resolvent(self,
                                     p,
                                     alpha=1,
                                     pmlbegin=None,
                                     pmlend=None,
                                     deg=2,
                                     inverse='umfpack',
                                     autoupdate=True):
        """
        Make the matrices needed for formulating the vector
        leaky mode eigensystem with frequency-independent
        C² PML map mapped_x = x * (1 + 1j * α * φ(r)) where
        φ is a C² function of the radius r.
        Using the resolvent T = A - C * D⁻¹ * B.
        INPUTS:
        * p: polynomial degree of finite elements
        * alpha: PML strength
        * pmlbegin: radius where PML begins
        * pmlend: radius where PML ends
        * deg: degree of the PML polynomial
        * inverse: inverse method to use in spectral projector
        * autoupdate: whether to use autoupdate in NGSolve
        OUTPUTS:
        * ResolventVectorMode: resolvent
        * m: bilinear form for the LHS
        * a, b, c, d: bilinear forms for the block matrices for the RHS
        * dinv: inverse of d
        """
        print('ModeSolver.smoothvecpmlsystem_resolvent called...\n')
        # raise NotImplementedError('This is not working yet.')
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with smooth.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout

        if self.gamma is None:
            self.set_vecpml_coeff(alpha, pmlbegin, pmlend, maxderiv=3)

        # Get symbolic functions
        detj = self.detj
        kappa = self.kappa
        gamma = self.gamma

        # Make linear eigensystem, cf. self.vecmodesystem
        n2 = self.index * self.index
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True,
                     autoupdate=autoupdate)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True,
                  autoupdate=autoupdate)

        E, F = X.TnT()
        phi, psi = Y.TnT()

        m = ng.BilinearForm(X)
        a = ng.BilinearForm(X)
        c = ng.BilinearForm(trialspace=Y, testspace=X)
        b = ng.BilinearForm(trialspace=X, testspace=Y)
        # d = ng.BilinearForm(Y)
        d = ng.BilinearForm(Y, condense=True)

        m += (gamma * E) * F * dx
        a += ((kappa * curl(E)) * curl(F) + self.V * (gamma * E) * F) * dx
        c += (gamma * grad(phi)) * F * dx
        b += n2 * (gamma * E) * grad(psi) * dx
        d += -n2 * detj * phi * psi * dx

        with ng.TaskManager():
            try:
                m.Assemble()
                a.Assemble()
                c.Assemble()
                b.Assemble()
                d.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                m.Assemble()
                a.Assemble()
                c.Assemble()
                b.Assemble()
                d.Assemble()
            res, dinv = self.make_resolvent_maxwell(m,
                                                    a,
                                                    b,
                                                    c,
                                                    d,
                                                    X,
                                                    Y,
                                                    inverse=inverse,
                                                    autoupdate=autoupdate)

        return res, m, a, b, c, d, dinv

    # ###################################################################
    # # LEAKY MODES #####################################################

    def leakymode_smooth(self,
                         p,
                         radiusZ2=0.1,
                         centerZ2=4,
                         pmlbegin=None,
                         pmlend=None,
                         alpha=1,
                         npts=8,
                         nspan=5,
                         seed=1,
                         within=None,
                         rhoinv=0.0,
                         quadrule='circ_trapez_shift',
                         inverse='umfpack',
                         verbose=True,
                         **feastkwargs):
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

        a, b, X = self.smoothpmlsystem(p,
                                       alpha=alpha,
                                       pmlbegin=pmlbegin,
                                       pmlend=pmlend)
        # OMIT m computation

        P = SpectralProjNG(X,
                           a.mat,
                           b.mat,
                           radius=radiusZ2,
                           center=centerZ2,
                           npts=npts,
                           checks=False,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           verbose=verbose,
                           inverse=inverse)

        Y = NGvecs(X, nspan)
        Yl = NGvecs(X, nspan)
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        zsqr, Y, history, Yl = P.feast(Y,
                                       Yl=Yl,
                                       hermitian=False,
                                       **feastkwargs)
        ewhist, cgd = history[-2], history[-1]

        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        bdrnrm = self.boundarynorm(Y)
        if np.max(bdrnrm) > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        moreoutputs = {
            'ewshistory': ewhist,
            'bdrnorm': bdrnrm,
            'converged': cgd
        }

        return zsqr, Y, Yl, beta, P, moreoutputs

    def leakyvecmodes_smooth_compound(self,
                                      p=None,
                                      radius=None,
                                      center=None,
                                      pmlbegin=None,
                                      pmlend=None,
                                      alpha=None,
                                      npts=None,
                                      nspan=None,
                                      seed=1,
                                      within=None,
                                      rhoinv=0.0,
                                      quadrule='circ_trapez_shift',
                                      inverse='umfpack',
                                      verbose=True,
                                      **feastkwargs):
        """
        Compute vector leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Using the compound finite element space X*Y and compound
        bilinear forms.

        Inputs and outputs are as documented in leakymode_auto(...). The
        only difference is that here you may override the starting and
        ending radius of PML by providing pmlbegin, pmlend.
        """
        print('ModeSolver.leakyvecmodes_smooth_compound called on:\n', self)
        # Check validity of inputs
        if p is None or radius is None or center is None:
            raise ValueError('Missing input(s)')
        # Get compound system
        aa, mm, Z = self.smoothvecpmlsystem_compound(p,
                                                     alpha=alpha,
                                                     pmlbegin=pmlbegin,
                                                     pmlend=pmlend,
                                                     deg=2,
                                                     autoupdate=True)
        # Create spectral projector
        P = SpectralProjNG(Z,
                           aa.mat,
                           mm.mat,
                           radius=radius,
                           center=center,
                           npts=npts,
                           checks=False,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           verbose=verbose,
                           inverse=inverse)

        # Set up NGvecs
        E_phi_r = NGvecs(Z, nspan)
        E_phi_l = NGvecs(Z, nspan)
        E_phi_r.setrandom(seed=seed)
        E_phi_l.setrandom(seed=seed)

        # Use FEAST
        zsqr, E_phi_r, history, E_phi_l = P.feast(E_phi_r,
                                                  Yl=E_phi_l,
                                                  hermitian=False,
                                                  **feastkwargs)

        # Compute betas, extract relevant variables
        ewhist, cgd = history[-2], history[-1]
        beta = self.betafrom(zsqr)

        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        # Unpack E_phi_r, E_phi_l into E_r, E_l, phi_r, phi_l
        X, Y = Z.components
        E_r = NGvecs(X, E_phi_r.m)
        E_l = NGvecs(X, E_phi_l.m)
        phi_r = NGvecs(Y, E_phi_r.m)
        phi_l = NGvecs(Y, E_phi_l.m)

        for i in range(E_phi_r.m):
            E_r._mv[i].data = E_phi_r[i].components[0].vec.data
            E_l._mv[i].data = E_phi_l[i].components[0].vec.data
            phi_r._mv[i].data = E_phi_r[i].components[1].vec.data
            phi_l._mv[i].data = E_phi_l[i].components[1].vec.data

        maxbdrnrm_r = np.max(self.boundarynorm(E_r))
        maxbdrnrm_l = np.max(self.boundarynorm(E_l))
        maxbdrnrm = max(maxbdrnrm_r, maxbdrnrm_l)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        moreoutputs = {
            'ewshistory': ewhist,
            'bdrnorm': maxbdrnrm,
            'converged': cgd,
        }

        return zsqr, E_r, E_l, phi_r, phi_l, beta, P, moreoutputs

    def leakyvecmodes_smooth_resolvent(self,
                                       p=None,
                                       radius=None,
                                       center=None,
                                       pmlbegin=None,
                                       pmlend=None,
                                       alpha=None,
                                       npts=None,
                                       nspan=None,
                                       seed=1,
                                       within=None,
                                       rhoinv=0.0,
                                       quadrule='circ_trapez_shift',
                                       inverse='umfpack',
                                       verbose=True,
                                       **feastkwargs):
        """
        Compute vector leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Using the resolvent T = A - C * D⁻¹ * B.

        Inputs and outputs are as documented in leakymode_auto(...). The
        only difference is that here you may override the starting and
        ending radius of PML by providing pmlbegin, pmlend.
        """
        print('ModeSolver.leakyvecmodes_smooth_resolvent called on:\n', self)
        # Check validity of inputs
        if p is None or radius is None or center is None:
            raise ValueError('Missing input(s)')
        # Get compound system
        res, m, a, b, c, d, dinv = self.smoothvecpmlsystem_resolvent(
            p,
            alpha=alpha,
            pmlbegin=pmlbegin,
            pmlend=pmlend,
            deg=2,
            inverse=inverse,
            autoupdate=True)

        # Create spectral projector
        P = SpectralProjNGR(
            lambda z: res(z, self.V, self.index, inverse=inverse),
            radius=radius,
            center=center,
            npts=npts,
            checks=False,
            within=within,
            rhoinv=rhoinv,
            quadrule=quadrule,
            verbose=verbose,
            inverse=inverse)

        # Unpack spaces from resolvent
        X, Y = res.XY.components
        # Set up NGvecs
        E_r = NGvecs(X, nspan, M=m)
        E_l = NGvecs(X, nspan, M=m)
        E_r.setrandom(seed=seed)
        E_l.setrandom(seed=seed)

        print('Using FEAST to search for vector leaky modes in')
        print(f'circle of radius {radius} centered at {center}')
        print(f'assuming not more than {nspan} modes in this interval')
        print(f'System size: {E_r.n} x {E_r.n}  Inverse type: {inverse}')

        # Use FEAST
        zsqr, E_r, history, E_l = P.feast(E_r,
                                          Yl=E_l,
                                          hermitian=False,
                                          **feastkwargs)

        # Compute betas, extract relevant variables
        ewhist, cgd = history[-2], history[-1]
        beta = self.betafrom(zsqr)

        print(f'Results:\n\tZ²: {zsqr}')
        print(f'\tbeta: {beta}')
        print(f'\tCL dB/m: {20 * beta.imag / np.log(10)}')

        maxbdrnrm_r = np.max(self.boundarynorm(E_r))
        maxbdrnrm_l = np.max(self.boundarynorm(E_l))
        maxbdrnrm = max(maxbdrnrm_r, maxbdrnrm_l)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        moreoutputs = {
            'ewshistory': ewhist,
            'bdrnorm': maxbdrnrm,
            'converged': cgd,
        }

        # TODO Compute phi_r, phi_l

        return zsqr, E_r, E_l, beta, P, moreoutputs

    # ###################################################################
    # ERROR ESTIMATORS FOR ADAPTIVITY ###################################

    def eestimator_maxwell(self, rgt, lft, lam):
        """
        DWR error estimator for Maxwell eigenproblem in compound
        form. We write eta = eta_1 + eta_2 + eta_3, where
            eta_i = sqrt(Omega_i_R * Rho_i_R) + sqrt(Omega_i_L * Rho_i_L)
        INPUT:
        * lft: left eigenfunction as NGvecs object for the compound form
        * rgt: right eigenfunction as NGvecs object for the compound form
        * lam: eigenvalue
        OUTPUT:
        * Eta: element-wise error estimator
        * Etas: dictionary with more info (see code)
        """

        assert rgt.m == lft.m and len(lam) == rgt.m, \
            'Check FEAST output:\n' + f'rgt.m {rgt.m} != lft.m {lft.m}'

        if self.gamma is None:
            raise ValueError('PML coefficients not set. Use set_vecpml_coeff.')

        eta1s = []
        eta2s = []
        eta3s = []
        kappa = self.kappa
        gamma = self.gamma
        detj = self.detj

        h = ng.specialcf.mesh_size
        n = ng.specialcf.normal(self.mesh.dim)
        n2 = self.index * self.index
        W = ng.L2(self.mesh, order=self.p, complex=True)
        kcurlE = ng.GridFunction(W)
        W2 = ng.VectorL2(self.mesh, order=self.p + 1, complex=True)
        flux = ng.GridFunction(W2)

        for i in range(rgt.m):

            R = rgt.gridfun('R', i=i)
            L = lft.gridfun('L', i=i)
            Z2 = lam[i]
            ER = R.components[0]
            EL = L.components[0]
            phiR = R.components[1]
            phiL = L.components[1]
            V = self.V

            with ng.TaskManager():

                kcurlE.Set(kappa * curl(ER))
                gradkcurlER = grad(kcurlE)
                rotkcurlER = CF((gradkcurlER[1], -gradkcurlER[0]))
                ggphiR = gamma * grad(phiR)
                rho1Ri = rotkcurlER + ggphiR + (V - Z2) * gamma * ER
                rho1Rj = kcurlE - kcurlE.Other()
                rho1Rint = h * h * InnerProduct(rho1Ri, rho1Ri)
                rho1Rjmp = 0.5 * h * rho1Rj * Conj(rho1Rj)
                Rho1R = Integrate(rho1Rint * dx +
                                  rho1Rjmp * dx(element_boundary=True),
                                  self.mesh,
                                  element_wise=True)

                flux.Set(ggphiR + (V - Z2) * gamma * ER)
                gradEphiR = Grad(flux)
                divEphiR = gradEphiR[0, 0] + gradEphiR[1, 1]
                rho2Rj = (flux - flux.Other()) * n
                rho2Rint = h * h * divEphiR * Conj(divEphiR)
                rho2Rjmp = 0.5 * h * rho2Rj * Conj(rho2Rj)
                Rho2R = Integrate(rho2Rint * dx +
                                  rho2Rjmp * dx(element_boundary=True),
                                  self.mesh,
                                  element_wise=True)

                flux.Set(n2 * gamma * ER)
                gradngER = Grad(flux)
                divngER = gradngER[0, 0] + gradngER[1, 1]
                rho3Ri = n2 * detj * phiR + divngER
                rho3Rj = (flux - flux.Other()) * n
                rho3Rint = h * h * rho3Ri * Conj(rho3Ri)
                rho3Rjmp = 0.5 * h * rho3Rj * Conj(rho3Rj)
                Rho3R = Integrate(rho3Rint * dx +
                                  rho3Rjmp * dx(element_boundary=True),
                                  self.mesh,
                                  element_wise=True)

                Omega1R = Integrate(InnerProduct(curl(ER), curl(ER)) * dx,
                                    self.mesh,
                                    element_wise=True)
                Omega2R = Integrate(InnerProduct(ER, ER) * dx,
                                    self.mesh,
                                    element_wise=True)
                Omega3R = Integrate(InnerProduct(grad(phiR), grad(phiR)) * dx,
                                    self.mesh,
                                    element_wise=True)

                kcurlE.Set(Conj(kappa) * curl(EL))
                gradkcurlEL = grad(kcurlE)
                rotkcurlEL = CF((gradkcurlEL[1], -gradkcurlEL[0]))
                rho1Li = rotkcurlEL + n2 * Conj(gamma) * grad(phiL) + \
                    Conj((V - Z2) * gamma) * EL
                rho1Lj = kcurlE - kcurlE.Other()
                rho1Lint = h * h * InnerProduct(rho1Li, rho1Li)
                rho1Ljmp = 0.5 * h * rho1Lj * Conj(rho1Lj)
                Rho1L = Integrate(rho1Lint * dx +
                                  rho1Ljmp * dx(element_boundary=True),
                                  self.mesh,
                                  element_wise=True)

                flux.Set(n2 * Conj(gamma) * grad(phiL) +
                         Conj((V - Z2) * gamma) * EL)
                gradEphiL = Grad(flux)
                divEphiL = gradEphiL[0, 0] + gradEphiL[1, 1]
                rho2Lj = (flux - flux.Other()) * n
                rho2Lint = h * h * divEphiL * Conj(divEphiL)
                rho2Ljmp = 0.5 * h * rho2Lj * Conj(rho2Lj)
                Rho2L = Integrate(rho2Lint * dx +
                                  rho2Ljmp * dx(element_boundary=True),
                                  self.mesh,
                                  element_wise=True)

                flux.Set(Conj(gamma) * EL)
                gradgEL = Grad(flux)
                divgEL = gradgEL[0, 0] + gradgEL[1, 1]
                rho3Li = n2 * Conj(detj) * phiL + divgEL
                rho3Lj = (flux - flux.Other()) * n
                rho3Lint = h * h * rho3Li * Conj(rho3Li)
                rho3Ljmp = 0.5 * h * rho3Lj * Conj(rho3Lj)
                Rho3L = Integrate(rho3Lint * dx +
                                  rho3Ljmp * dx(element_boundary=True),
                                  self.mesh,
                                  element_wise=True)

                Omega1L = Integrate(InnerProduct(curl(EL), curl(EL)) * dx,
                                    self.mesh,
                                    element_wise=True)
                Omega2L = Integrate(InnerProduct(EL, EL) * dx,
                                    self.mesh,
                                    element_wise=True)
                Omega3L = Integrate(InnerProduct(grad(phiL), grad(phiL)) * dx,
                                    self.mesh,
                                    element_wise=True)

                Eta1 = np.sqrt(Omega1L.real.NumPy() * Rho1R.real.NumPy())
                Eta1 += np.sqrt(Omega1R.real.NumPy() * Rho1L.real.NumPy())
                eta1s.append(Eta1)

                Eta2 = np.sqrt(Omega2L.real.NumPy() * Rho2R.real.NumPy())
                Eta2 += np.sqrt(Omega2R.real.NumPy() * Rho2L.real.NumPy())
                eta2s.append(Eta2)

                Eta3 = np.sqrt(Omega3L.real.NumPy() * Rho3R.real.NumPy())
                Eta3 += np.sqrt(Omega3R.real.NumPy() * Rho3L.real.NumPy())
                eta3s.append(Eta3)

        Eta = np.zeros_like(eta1s[0])
        Eta1 = np.zeros_like(Eta)
        Eta2 = np.zeros_like(Eta)
        Eta3 = np.zeros_like(Eta)
        for i in range(rgt.m):
            Eta1 += eta1s[i]
            Eta2 += eta2s[i]
            Eta3 += eta3s[i]
        Eta = Eta1 + Eta2 + Eta3

        Etas = {
            'eta1s': eta1s,
            'eta2s': eta2s,
            'eta3s': eta3s,
            'Eta1': (Eta1, np.max(Eta1)),
            'Eta2': (Eta2, np.max(Eta2)),
            'Eta3': (Eta3, np.max(Eta3)),
        }

        return Eta, Etas

    def __compute_avr_ngvecs(self, gf, name=None):
        """
        Return a grid function that takes the average of the components
        of the (multi-component) grid function gf, provided in NGvecs
        format.
        OBS: This function might be redundant, as we are not implementing
        the averaged error estimator, but a list of error estimators.
        INPUT:
        * gf: NGvecs object
        * name: name of the new grid function
        OUTPUT:
        * new_gf: GridFunction object
        """
        new_gf = ng.GridFunction(gf.fes,
                                 name=name +
                                 ' avr' if name is not None else 'avr',
                                 autoupdate=gf.fes.autoupdate)
        for i in range(gf.m):
            new_gf.vec.data += gf[i].vec.data
        new_gf.vec.data /= gf.m
        return new_gf

    def eestimator_helmholtz(self, rgt, lft, lam, A, B, V):
        """
        DWR error estimator for eigenvalues

        INPUT:
        * lft: left eigenfunction as NGvecs object
        * rgt: right eigenfunction as NGvecs object
        * lam: eigenvalue
        * A, B, V are such that the eigenproblem is
          -div(A grad u) + V B u = lam B  u

        OUTPUT:
        * ee: element-wise error estimator
        """
        assert rgt.m == lft.m, 'Check FEAST output:\n' + \
            f'rgt.m {rgt.m} != lft.m {lft.m}'
        if rgt.m > 1 or lft.m > 1:
            print('Taking average of multiple eigenfunctions')

        R = self.__compute_avr_ngvecs(rgt)  # rgt.gridfun('R', i=0)
        L = self.__compute_avr_ngvecs(lft)  # lft.gridfun('L', i=0)
        h = ng.specialcf.mesh_size
        n = ng.specialcf.normal(self.mesh.dim)

        AgradR = A * grad(R)
        divAgradR = AgradR[0].Diff(ng.x) + AgradR[1].Diff(ng.y)
        AgradL = A * grad(L)
        divAgradL = AgradL[0].Diff(ng.x) + AgradL[1].Diff(ng.y)

        r = h * (divAgradR - V * B * R + lam * R)
        rhoR = Integrate(InnerProduct(r, r) * dx, self.mesh, element_wise=True)
        r = h * (divAgradL - V * B * L + np.conj(lam) * L)
        rhoL = Integrate(InnerProduct(r, r) * dx, self.mesh, element_wise=True)
        jR = n * (AgradR - AgradR.Other())
        jL = n * (AgradL - AgradL.Other())
        rhoR += Integrate(0.5 * h * InnerProduct(jR, jR) *
                          dx(element_boundary=True),
                          self.mesh,
                          element_wise=True)
        rhoL += Integrate(0.5 * h * InnerProduct(jL, jL) *
                          dx(element_boundary=True),
                          self.mesh,
                          element_wise=True)

        # left as a reminder
        # def hess(gf):
        #     return gf.Operator("hesse")
        # omegaR = Integrate(h * h * InnerProduct(hess(R), hess(R)),
        #                    self.mesh,
        #                    element_wise=True)
        # omegaL = Integrate(h * h * InnerProduct(hess(L), hess(L)),
        #                    self.mesh,
        #                    element_wise=True)

        omegaR = Integrate(h * InnerProduct(grad(R), grad(R)),
                           self.mesh,
                           element_wise=True)
        omegaL = Integrate(h * InnerProduct(grad(L), grad(L)),
                           self.mesh,
                           element_wise=True)

        ee = np.sqrt(omegaR.real.NumPy() * rhoR.real.NumPy())
        ee += np.sqrt(omegaL.real.NumPy() * rhoL.real.NumPy())

        return ee

    def eestimator_maxwell_compound(self, rgt, lft, lam):
        """
        DWR error estimator for eigenvalues Maxwell eigenproblem for compound
        form. We write eta = eta_1 + eta_2 + eta_3, where
            eta_i = sqrt(Omega_i_R * Rho_i_R) + sqrt(Omega_i_L * Rho_i_L)
        INPUT:
        * lft: left eigenfunction as NGvecs object for the compound form
        * rgt: right eigenfunction as NGvecs object for the compound form
        * lam: eigenvalue
        OUTPUT:
        * Eta: element-wise error estimator
        * Etas: dictionary with three tuples of error estimators and
                    their maximum component
        """
        assert rgt.m == lft.m, 'Check FEAST output:\n' + \
            f'rgt.m {rgt.m} != lft.m {lft.m}'
        if rgt.m > 1 or lft.m > 1:
            print('Taking average of multiple eigenfunctions')

        if self.gamma is None:
            raise ValueError('PML coefficients not set. Use set_vecpml_coeff.'
                             ' Check code logic.')
        # Extract coefficient functions
        kappa = self.kappa
        # kappa_conj = self.kappa_conj
        gamma = self.gamma
        # gamma_conj = self.gamma_conj
        detj = self.detj
        detj_conj = self.detj_conj

        h = ng.specialcf.mesh_size
        n = ng.specialcf.normal(self.mesh.dim)
        n2 = self.index * self.index

        # Extract grid functions and its components
        rgt_avg = self.__compute_avr_ngvecs(rgt, name='rgt')
        lft_avg = self.__compute_avr_ngvecs(lft, name='lft')
        E_r = rgt_avg.components[0]
        E_l = lft_avg.components[0]
        phi_r = rgt_avg.components[1]
        phi_l = lft_avg.components[1]

        # Compute derivatives for residuals and weights
        V = self.V
        # gamma.E -> divfree, n2.gamma.E
        gE_r = gamma * E_r
        # gE_l = gamma_conj * E_l
        gE_l = gamma * ng.Conj(E_l)
        # curl.E -> kappa.curl.E, w1
        curlE_r = curl(E_r)
        curlE_l = curl(E_l)
        # kappa.curl.E -> rot.kappa.curl.E, jump.kappa.curl.E
        kcurlE_r = kappa * curlE_r
        # kcurlE_l = kappa_conj * curlE_l
        kcurlE_l = kappa * ng.Conj(curlE_l)
        # rot.kappa.curl.E.xy -> r1
        rotkcurlE_r_x = kcurlE_r.Diff(ng.y)
        rotkcurlE_r_y = -kcurlE_r.Diff(ng.x)
        # rotkcurlE_l_x = kcurlE_l.Diff(ng.y)
        # rotkcurlE_l_y = -kcurlE_l.Diff(ng.x)
        rotkcurlE_l_x = ng.Conj(kcurlE_l.Diff(ng.y))
        rotkcurlE_l_y = ng.Conj(-kcurlE_l.Diff(ng.x))
        # jump.kappa.curl.E -> r1
        j_kcurlE_r = kcurlE_r - kcurlE_r.Other()
        j_kcurlE_l = kcurlE_l - kcurlE_l.Other()
        # gamma.grad(phi) -> gamma.grad.phi, w3
        gphi_r = grad(phi_r)
        gphi_l = grad(phi_l)
        # gamma.grad.phi -> divfree
        ggphi_r = gamma * gphi_r
        # ggphi_l = n2 * gamma_conj * gphi_l
        ggphi_l = n2 * gamma * ng.Conj(gphi_l)
        # divfree -> r1, div, jump.divfree
        divfree_r = ggphi_r + (V - lam) * gE_r
        # divfree_l = ggphi_l + (V - np.conj(lam)) * gE_l
        divfree_l = ggphi_l + (V - lam) * gE_l
        # div -> r2
        div_r = divfree_r[0].Diff(ng.x) + divfree_r[1].Diff(ng.y)
        # div_l = divfree_l.Diff(ng.x) + divfree_l.Diff(ng.y)
        div_l = ng.Conj(divfree_l[0].Diff(ng.x) + divfree_l[1].Diff(ng.y))
        # jump.divfree -> r2
        j_divfree_r = n * (divfree_r - divfree_r.Other())
        j_divfree_l = n * (divfree_l - divfree_l.Other())
        # n2.gamma.E -> div.E, jump.n2.gamma.E
        nE_r = n2 * gE_r
        nE_l = gE_l
        # jump.n2.gamma.E -> r3
        j_nE_r = n * (nE_r - nE_r.Other())
        j_nE_l = n * (nE_l - nE_l.Other())
        # n2.detj.phi -> r3
        nphi_r = n2 * detj * phi_r
        nphi_l = n2 * detj_conj * phi_l
        # div.E -> r3
        divE_r = nE_r[0].Diff(ng.x) + nE_r[1].Diff(ng.y)
        divE_l = nE_l[0].Diff(ng.x) + nE_l[1].Diff(ng.y)

        # Residuals and weights (as integrands)
        # # Residuals
        rho1_r_x = h * (rotkcurlE_r_x + divfree_r[0])
        rho1_r_y = h * (rotkcurlE_r_y + divfree_r[1])
        rho1_r_j = ng.sqrt(0.5 * h) * j_kcurlE_r

        rho1_l_x = h * (rotkcurlE_l_x + divfree_l[0])
        rho1_l_y = h * (rotkcurlE_l_y + divfree_l[1])
        rho1_l_j = ng.sqrt(0.5 * h) * j_kcurlE_l

        rho2_r = h * div_r
        rho2_r_j = ng.sqrt(0.5 * h) * j_divfree_r
        rho2_l = h * div_l
        rho2_l_j = ng.sqrt(0.5 * h) * j_divfree_l

        rho3_r = h * (divE_r + nphi_r)
        rho3_l = h * (divE_l + nphi_l)
        rho3_r_j = ng.sqrt(0.5 * h) * j_nE_r
        rho3_l_j = ng.sqrt(0.5 * h) * j_nE_l

        # # Weights
        omega1_r = curlE_l
        omega1_l = curlE_r

        omega2_r = E_l
        omega2_l = E_r

        omega3_r = gphi_l
        omega3_l = gphi_r

        # Residuals and weights (as numpy arrays)
        Rho1_r = Integrate(
            (InnerProduct(rho1_r_x, rho1_r_x) +
             InnerProduct(rho1_r_y, rho1_r_y)) * dx +
            InnerProduct(rho1_r_j, rho1_r_j) * dx(element_boundary=True),
            self.mesh,
            element_wise=True)

        Rho2_r = Integrate(
            InnerProduct(rho2_r, rho2_r) * dx +
            InnerProduct(rho2_r_j, rho2_r_j) * dx(element_boundary=True),
            self.mesh,
            element_wise=True)

        Rho3_r = Integrate(
            InnerProduct(rho3_r, rho3_r) * dx +
            InnerProduct(rho3_r_j, rho3_r_j) * dx(element_boundary=True),
            self.mesh,
            element_wise=True)

        Rho1_l = Integrate(
            (InnerProduct(rho1_l_x, rho1_l_x) +
             InnerProduct(rho1_l_y, rho1_l_y)) * dx +
            InnerProduct(rho1_l_j, rho1_l_j) * dx(element_boundary=True),
            self.mesh,
            element_wise=True)

        Rho2_l = Integrate(
            InnerProduct(rho2_l, rho2_l) * dx +
            InnerProduct(rho2_l_j, rho2_l_j) * dx(element_boundary=True),
            self.mesh,
            element_wise=True)

        Rho3_l = Integrate(
            InnerProduct(rho3_l, rho3_l) * dx +
            InnerProduct(rho3_l_j, rho3_l_j) * dx(element_boundary=True),
            self.mesh,
            element_wise=True)

        Omega1_r = Integrate(InnerProduct(omega1_r, omega1_r) * dx,
                             self.mesh,
                             element_wise=True)

        Omega2_r = Integrate(InnerProduct(omega2_r, omega2_r) * dx,
                             self.mesh,
                             element_wise=True)

        Omega3_r = Integrate(InnerProduct(omega3_r, omega3_r) * dx,
                             self.mesh,
                             element_wise=True)

        Omega1_l = Integrate(InnerProduct(omega1_l, omega1_l) * dx,
                             self.mesh,
                             element_wise=True)

        Omega2_l = Integrate(InnerProduct(omega2_l, omega2_l) * dx,
                             self.mesh,
                             element_wise=True)

        Omega3_l = Integrate(InnerProduct(omega3_l, omega3_l) * dx,
                             self.mesh,
                             element_wise=True)

        Eta1 = np.sqrt(Omega1_r.real.NumPy() * Rho1_r.real.NumPy())
        Eta1 += np.sqrt(Omega1_l.real.NumPy() * Rho1_l.real.NumPy())

        Eta2 = np.sqrt(Omega2_r.real.NumPy() * Rho2_r.real.NumPy())
        Eta2 += np.sqrt(Omega2_l.real.NumPy() * Rho2_l.real.NumPy())

        Eta3 = np.sqrt(Omega3_r.real.NumPy() * Rho3_r.real.NumPy())
        Eta3 += np.sqrt(Omega3_l.real.NumPy() * Rho3_l.real.NumPy())

        Eta = Eta1 + Eta2 + Eta3

        Etas = {
            'Eta1': (Eta1, np.max(Eta1)),
            'Eta2': (Eta2, np.max(Eta2)),
            'Eta3': (Eta3, np.max(Eta3)),
        }

        return Eta, Etas

    def eestimator_maxwell_resolvent(self, Er, El, phir, phil, lam):
        """
        DWR error estimator for eigenvalues
        Maxwell eigenproblem for resolvent form
        INPUT:
        * Er: right eigenfunction, transversal component as NGvecs object
        * El: left eigenfunction, transversal component as NGvecs object
        * phir: right eigenfunction, longitudinal component as NGvecs object
        * phil: left eigenfunction, longitudinal component as NGvecs object
        * lam: eigenvalue
        OUTPUT:
        * ee: element-wise error estimator
        """
        raise NotImplementedError('Not working yet. Need to be adapted.')
        assert Er.m == El.m and phir.m == phil.m, 'Check FEAST output:\n' + \
            f'Er.m {Er.m} != El.m {El.m} or phir.m {phir.m} != phil.m {phil.m}'
        if Er.m > 1 or El.m > 1 or phir.m > 1 or phil.m > 1:
            print('Taking average of multiple eigenfunctions')

        # Extract coefficient functions
        mu = self.mu
        eta = self.eta
        mu_dr = self.mu_dr
        eta_dr = self.eta_dr
        # jac = self.jac
        jacinv = self.jacinv
        detj = self.detj

        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)

        h = ng.specialcf.mesh_size
        n = ng.specialcf.normal(self.mesh.dim)
        n2 = self.index * self.index
        kappa = (mu / eta_dr**3) * (1 + (mu_dr * r**2) / eta)**2

        # Extract grid functions and its components
        E_r = self.__compute_avr_ngvecs(Er, name='E_r')
        E_l = self.__compute_avr_ngvecs(El, name='E_l')
        phi_r = self.__compute_avr_ngvecs(phir, name='phi_r')
        phi_l = self.__compute_avr_ngvecs(phil, name='phi_l')

        # Compute coefficients
        # # Mapped fields
        JE_r = detj * jacinv * jacinv * E_r
        JE_l = detj * jacinv * jacinv * E_l
        # # First order derivatives
        gradE_r = grad(E_r)
        gradphi_r = grad(phi_r)
        gradE_l = grad(E_l)
        gradphi_l = grad(phi_l)
        # # PML First order derivatives
        Jgradphi_r = detj * jacinv * jacinv * gradphi_r
        Jgradphi_l = detj * jacinv * jacinv * gradphi_l
        # # Remaining terms
        curlE_r = kappa * curl(E_r)
        rotcurlE_r_x = curlE_r.Diff(y)
        rotcurlE_r_y = -curlE_r.Diff(x)

        curlE_l = kappa * curl(E_l)
        rotcurlE_l_x = curlE_l.Diff(y)
        rotcurlE_l_y = -curlE_l.Diff(x)

        div_E_r = JE_r[0].Diff(x) + JE_r[1].Diff(y)
        div_E_l = JE_l[0].Diff(x) + JE_l[1].Diff(y)

        # Residual integrals
        # j is for jump, r is for right, l is for left
        # t is for transversal, z is for longitudinal
        r_t_x = h * (rotcurlE_r_x + self.V * JE_r[0] + Jgradphi_r[0] -
                     lam * JE_r[0])
        r_t_y = h * (rotcurlE_r_y + self.V * JE_r[1] + Jgradphi_r[1] -
                     lam * JE_r[1])
        jr_t = curlE_r - curlE_r.Other()

        r_z = h * (n2 * div_E_r + n2 * detj * phi_r)
        jr_z = n2 * n * (JE_r - JE_r.Other())

        l_t_x = h * (rotcurlE_l_x + self.V * JE_l[0] + n2 * Jgradphi_l[0] -
                     np.conj(lam) * JE_l[0])
        l_t_y = h * (rotcurlE_l_y + self.V * JE_l[1] + n2 * Jgradphi_l[1] -
                     np.conj(lam) * JE_l[1])
        jl_t = curlE_l - curlE_l.Other()

        l_z = h * (div_E_l + n2 * detj * phi_l)
        jl_z = n2 * n * (JE_l - JE_l.Other())

        rho_r = Integrate(InnerProduct(r_t_x, r_t_x) * dx,
                          self.mesh,
                          element_wise=True)
        rho_r += Integrate(InnerProduct(r_t_y, r_t_y) * dx,
                           self.mesh,
                           element_wise=True)
        rho_r += Integrate(0.5 * h * InnerProduct(jr_t, jr_t) *
                           dx(element_boundary=True),
                           self.mesh,
                           element_wise=True)
        rho_r += Integrate(InnerProduct(r_z, r_z) * dx,
                           self.mesh,
                           element_wise=True)
        rho_r += Integrate(0.5 * h * InnerProduct(jr_z, jr_z) *
                           dx(element_boundary=True),
                           self.mesh,
                           element_wise=True)

        rho_l = Integrate(InnerProduct(l_t_x, l_t_x) * dx,
                          self.mesh,
                          element_wise=True)
        rho_l += Integrate(InnerProduct(l_t_y, l_t_y) * dx,
                           self.mesh,
                           element_wise=True)
        rho_l += Integrate(0.5 * h * InnerProduct(jl_t, jl_t) *
                           dx(element_boundary=True),
                           self.mesh,
                           element_wise=True)
        rho_l += Integrate(InnerProduct(l_z, l_z) * dx,
                           self.mesh,
                           element_wise=True)
        rho_l += Integrate(0.5 * h * InnerProduct(jl_z, jl_z) *
                           dx(element_boundary=True),
                           self.mesh,
                           element_wise=True)

        omega_r = Integrate(InnerProduct(gradE_r, gradE_r),
                            self.mesh,
                            element_wise=True)
        omega_r += Integrate(InnerProduct(gradphi_r, gradphi_r),
                             self.mesh,
                             element_wise=True)

        omega_l = Integrate(InnerProduct(gradE_l, gradE_l),
                            self.mesh,
                            element_wise=True)
        omega_l += Integrate(InnerProduct(gradphi_l, gradphi_l),
                             self.mesh,
                             element_wise=True)

        ee = np.sqrt(omega_r.real.NumPy() * rho_r.real.NumPy())
        ee += np.sqrt(omega_l.real.NumPy() * rho_l.real.NumPy())

        return ee

    def leakymode_adapt(
            self,
            p,
            radiusZ2=0.1,
            centerZ2=4,
            maxndofs=200000,  # Stop if ndofs become larger than this
            visualize=True,  # Pause adaptive loop to see iterate
            pmlbegin=None,
            pmlend=None,
            alpha=10,
            npts=4,
            nspan=5,
            seed=1,
            within=None,
            rhoinv=0.0,
            quadrule='circ_trapez_shift',
            inverse='umfpack',
            trustme=False,
            verbose=True,
            **feastkwargs):
        """
        Compute leaky modes by DWR adaptivity, solving in each iteration a
        linear eigenproblem obtained using the (frequency-independent)
        C² smooth PML in which mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r.  The eigenproblem is
        solved by a non-selfadjoint FEAST algorithm.

        INPUT:

        * radiusZ2, centerZ2:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * maxndofs: Stop adaptive loop if number of dofs exceed this.
        * trustme: If True, modify the new centerZ2 to be the average
            of the Z² values of the modes found in the previous iteration.
        * visualize: If true, then pause adaptivity loop to see each iterate.
        * Remaining inputs are as documented in leakymode(..).

        OUTPUT:   zsqr, Yr, Yl, P

        * zsqr: Computed Z² at the finest mesh found by adaptivity.
        * Yl, Y: Corresponding left and right eigenspans.
        * P: spectral projector object that conducted FEAST.
        """

        ndofs = [0]
        Zsqrs = []
        if visualize:
            eevis = ng.GridFunction(ng.L2(self.mesh, order=0, autoupdate=True),
                                    name='estimator',
                                    autoupdate=True)
            ng.Draw(eevis)

        while ndofs[-1] < maxndofs:  # ADAPTIVITY LOOP ------------------
            a, b, X = self.smoothpmlsystem(p,
                                           alpha=alpha,
                                           autoupdate=True,
                                           pmlbegin=pmlbegin,
                                           pmlend=pmlend)

            Yr = NGvecs(X, nspan)
            Yl = NGvecs(X, nspan)
            Yr.setrandom(seed=seed)
            Yl.setrandom(seed=seed)

            # 1. SOLVE

            with ng.TaskManager():
                try:
                    a.Assemble()
                    b.Assemble()
                except Exception:
                    print('*** Trying again with larger heap')
                    ng.SetHeapSize(int(1e9))
                    a.Assemble()
                    b.Assemble()

            P = SpectralProjNG(X,
                               a.mat,
                               b.mat,
                               radius=radiusZ2,
                               center=centerZ2,
                               npts=npts,
                               within=within,
                               rhoinv=rhoinv,
                               checks=False,
                               quadrule=quadrule,
                               verbose=verbose,
                               inverse=inverse)

            zsqr, Yr, history, Yl = P.feast(Yr,
                                            Yl=Yl,
                                            hermitian=False,
                                            **feastkwargs)
            _, cgd = history[-2], history[-1]
            if not cgd:
                raise NotImplementedError('What to do when FEAST fails?')

            ndofs.append(Yr.fes.ndof)
            Zsqrs.append(zsqr)
            print(f'ADAPTIVITY at {ndofs[-1]:7d} ndofs: ' +
                  f'Zsqr = {Zsqrs[-1][0]:+10.8f}')

            # 2. ESTIMATE

            avr_zsqr = np.average(zsqr)
            ee = self.eestimator_helmholtz(Yr, Yl, avr_zsqr, self.pml_A,
                                           self.pml_B, self.V)
            if visualize:
                eevis.vec.FV().NumPy()[:] = ee
                ng.Draw(eevis)
                Yl.draw(name='LftEig')
                Yr.draw(name='RgtEig')
                input('* Pausing for visualization. Enter any key to continue')
            if ndofs[-1] > maxndofs:
                break

            # 3. MARK

            strat = AdaptivityStrategy(Strategy.AVG)
            strat.apply(self.mesh, ee)

            # 4. REFINE

            self.mesh.Refine()
            ngmesh = self.mesh.ngmesh.Copy()
            self.mesh = ng.Mesh(ngmesh)
            self.mesh.Curve(8)
            if trustme:
                centerZ2 = avr_zsqr
                npts = 1
                nspan = 1

        # Adaptivity loop done ------------------------------------------

        beta = self.betafrom(zsqr)
        print('Results:\n\tZ²:', zsqr)
        print('\tbeta:', beta)
        print('\tCL dB/m:', 20 * beta.imag / np.log(10))
        maxbdrnrm = np.max(self.boundarynorm(Yr))
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return Zsqrs, ndofs, Yr, Yl, beta, P

    def leakyvecmodes_adapt(self,
                            p,
                            radius,
                            center,
                            alpha=None,
                            pmlbegin=None,
                            pmlend=None,
                            maxndofs=200000,
                            visualize=True,
                            markfraction=0.1,
                            autoupdate=False,
                            trustme=True,
                            npts=4,
                            nspan=5,
                            seed=1,
                            within=None,
                            rhoinv=0.0,
                            quadrule='circ_trapez_shift',
                            inverse='umfpack',
                            verbose=True,
                            **feastkwargs):
        """
        Compute vector leaky modes by DWR adaptivity, solving in each
        iteration a linear eigenproblem obtained using the
        (frequency-independent) C² smooth PML in which
            mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r.  The eigenproblem is
        solved by a non-selfadjoint FEAST algorithm.

        INPUT:

        * radius, center:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * markfraction: if eta_T > markfraction * max_T eta_T,
            then mark element T for refinement. Here eta_T is the DWR
            error estimator.
        * maxndofs: Stop adaptive loop if number of dofs exceed this.
        * visualize: If true, then pause adaptivity loop to see each iterate.
        * autoupdate: If True, use NGSolve's autoupdate-on-refinement feature
            for meshes, spaces, and gridfunctions. (Unfortunately, as of
            May 2024, random segfaults are created by ngsolve autoupdate).
            If False, then after each adaptive refinement, copy the mesh,
            create new gridfunctions, new spaces, etc., in each iteration.
        * trustme: If True, then abandon contour checking, lock nspan to
            first converged dimension, and just do shifted inverse interation
            with shift set to mean of prior converged eigenvalue iterates.
        * Remaining inputs are as documented in leakymode(..).

        OUTPUT:   zsqr, eestimates, ER, EL, phiR, phiL, beta, P
        """

        ndofs = [0]
        Zsqrs = []
        errestimates = []
        checkcontour = 3
        if autoupdate and visualize:
            E = ng.L2(self.mesh, order=0, autoupdate=True)
            eevis = ng.GridFunction(E,
                                    name='estimator',
                                    autoupdate=True,
                                    nested=True)

        while ndofs[-1] < maxndofs:  # ADAPTIVITY LOOP ------------------

            aa, mm, Z = self.smoothvecpmlsystem_compound(p,
                                                         alpha=alpha,
                                                         pmlbegin=pmlbegin,
                                                         pmlend=pmlend,
                                                         autoupdate=autoupdate)
            uR = NGvecs(Z, nspan)
            uL = NGvecs(Z, nspan)
            uR.setrandom(seed=seed)
            uL.setrandom(seed=seed)
            print('ADAPTIVITY at ', uR.fes.ndof, ' ndofs:')
            print('  Assembling system...')

            # 1. SOLVE

            with ng.TaskManager():
                try:
                    aa.Assemble()
                    mm.Assemble()
                except Exception:
                    print('   *** Trying again with larger heap')
                    ng.SetHeapSize(int(1e9))
                    aa.Assemble()
                    mm.Assemble()

            P = SpectralProjNG(Z,
                               aa.mat,
                               mm.mat,
                               radius=radius,
                               center=center,
                               npts=npts,
                               within=within,
                               rhoinv=rhoinv,
                               checks=False,
                               quadrule=quadrule,
                               verbose=verbose,
                               inverse=inverse)
            zsqr, uR, history, uL = P.feast(uR,
                                            Yl=uL,
                                            hermitian=False,
                                            check_contour=checkcontour,
                                            **feastkwargs)
            _, cgd = history[-2], history[-1]
            if not cgd:
                raise ValueError('FEAST failed. Try another region')
            ndofs.append(uR.fes.ndof)
            Zsqrs.append(zsqr)
            print('  Computed eigenvalues:', zsqr)

            center = np.average(zsqr)
            if trustme:
                npts = 1
                nspan = len(zsqr)
                checkcontour = 0  # with this, radius is irrelevant

            # 2. ESTIMATE
            ee, more = self.eestimator_maxwell(uR, uL, zsqr)
            errestimates.append((sum(ee), more))
            print('  Error estimator:', errestimates[-1][0])

            if visualize:
                if autoupdate:
                    eevis.vec.FV().NumPy()[:] = ee
                else:
                    E = ng.L2(self.mesh, order=0)
                    eevis = ng.GridFunction(E, name='estimator')
                    eevis.vec.FV().NumPy()[:] = ee
                ng.Draw(eevis)
                ng.Draw(uR.gridfun(name="ER").components[0])
                ng.Draw(uR.gridfun(name="phiR").components[1])
                input('* Pausing for visualization. Enter any key to continue')

            if ndofs[-1] > maxndofs:
                break

            # 3. MARK
            maxee = np.max(ee)
            self.mesh.ngmesh.Elements2D().NumPy()["refine"] = \
                ee > markfraction * maxee
            nummarked = sum(self.mesh.ngmesh.Elements2D().NumPy()["refine"])
            print('  Marked ', nummarked, ' elements for refinement')

            # 4. REFINE

            self.mesh.Refine()
            if not autoupdate:
                ngmesh = self.mesh.ngmesh.Copy()
                self.mesh = ng.Mesh(ngmesh)
                self.mesh.Curve(8)

        # Adaptivity loop done ------------------------------------------

        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        # Unpack uR, uL into ER, EL, phiR, phiL
        X, Y = Z.components
        ER = NGvecs(X, uR.m)
        EL = NGvecs(X, uL.m)
        phiR = NGvecs(Y, uR.m)
        phiL = NGvecs(Y, uL.m)

        for i in range(uR.m):
            ER._mv[i].data = uR[i].components[0].vec.data
            EL._mv[i].data = uL[i].components[0].vec.data
            phiR._mv[i].data = uR[i].components[1].vec.data
            phiL._mv[i].data = uL[i].components[1].vec.data

        maxbdrnrm_r = np.max(self.boundarynorm(ER))
        maxbdrnrm_l = np.max(self.boundarynorm(EL))
        maxbdrnrm = max(maxbdrnrm_r, maxbdrnrm_l)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return Zsqrs, errestimates, ndofs, ER, EL, phiR, phiL, beta, P

    def leakyvecmodes_adapt_resolvent(
            self,
            p=None,
            radius=None,
            center=None,
            alpha=None,
            pmlbegin=None,
            pmlend=None,
            maxndofs=200000,  # Stop if ndofs become larger than this
            visualize=True,  # Pause adaptive loop to see iterate
            npts=4,
            nspan=5,
            seed=1,
            within=None,
            rhoinv=0.0,
            quadrule='circ_trapez_shift',
            inverse='umfpack',
            autoupdate=True,
            trustme=False,
            verbose=True,
            **feastkwargs):
        """
        Compute vector leaky modes by DWR adaptivity, solving in each
        iteration a linear eigenproblem obtained using the
        (frequency-independent) C² smooth PML in which
            mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r.  The eigenproblem is
        solved by a non-selfadjoint FEAST algorithm.
        We use the resolvent formulation of the eigenproblem., i.e.,
        we call smoothvecpmlsystem_resolvent instead of
        smoothvecpmlsystem_compound.

        INPUT:

        * radius, center:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * maxndofs: Stop adaptive loop if number of dofs exceed this.
        * trustme: If True, modify the new center to be the average
            of the Z² values of the modes found in the previous iteration.
        * visualize: If true, then pause adaptivity loop to see each iterate.
        * Remaining inputs are as documented in leakymode(..).

        OUTPUT:   zsqr, E_r, E_l, phi_r, phi_l, P
        """
        # Check validity of inputs
        if p is None or radius is None or center is None:
            raise ValueError('Missing input(s)')

        ndofs = [0]
        Zsqrs = []

        if visualize:
            eevis = ng.GridFunction(ng.L2(self.mesh, order=0, autoupdate=True),
                                    name='estimator',
                                    autoupdate=True,
                                    nested=True)
            ng.Draw(eevis)

        while ndofs[-1] < maxndofs:  # ADAPTIVITY LOOP ------------------

            res, m, a, b, c, d, dinv = self.smoothvecpmlsystem_resolvent(
                p,
                alpha=alpha,
                pmlbegin=pmlbegin,
                pmlend=pmlend,
                deg=2,
                inverse=inverse,
                autoupdate=autoupdate)
            X, Y = res.XY.components

            E_r = NGvecs(X, nspan, M=m.mat)
            E_l = NGvecs(X, nspan, M=m.mat)
            E_r.setrandom(seed=seed)
            E_l.setrandom(seed=seed)

            # 1. SOLVE

            # The assembling occurs in smoothvecpmlsystem_resolvent

            P = SpectralProjNGR(
                lambda z: res(z, self.V, self.index, inverse=inverse),
                radius=radius,
                center=center,
                npts=npts,
                within=within,
                rhoinv=rhoinv,
                quadrule=quadrule,
                verbose=verbose,
                inverse=inverse)

            zsqr, E_r, history, E_l = P.feast(E_r,
                                              Yl=E_l,
                                              hermitian=False,
                                              **feastkwargs)

            cgd = history[-1]
            if not cgd:
                raise NotImplementedError('What to do when FEAST fails?')
            # Implement average of multiple eigenfunctions
            avr_zsqr = np.average(zsqr)
            ndofs.append(E_r.fes.ndof)
            Zsqrs.append(zsqr)
            if E_r.m > 1:
                print(f'ADAPTIVITY at {ndofs[-1]:7d} ndofs: ' +
                      f'avg_zsqr = {avr_zsqr:+10.8f}')
            else:
                print(f'ADAPTIVITY at {ndofs[-1]:7d} ndofs: ' +
                      f'Zsqr = {Zsqrs[-1][0]:+10.8f}')

            # Get phi_r, phi_l from E_r, E_l
            phi_r, phi_l = self.__get_phi_from_E(E_r, E_l, b, c, d, dinv, Y)

            # 2. ESTIMATE
            ee = self.eestimator_maxwell_resolvent(E_r, E_l, phi_r, phi_l,
                                                   avr_zsqr)

            if visualize:
                eevis.vec.FV().NumPy()[:] = ee
                ng.Draw(eevis)
                for i in range(E_r.m):
                    E_r.draw(name="E_r" + str(i))
                    phi_r.draw(name="phi_r" + str(i))
                    E_l.draw(name="E_l" + str(i))
                    phi_l.draw(name="phi_l" + str(i))
                    input('* Pausing for visualization.'
                          'Enter any key to continue')

            if ndofs[-1] > maxndofs:
                break

            # 3. MARK

            strat = AdaptivityStrategy(Strategy.AVG)
            strat.apply(self.mesh, ee)

            # 4. REFINE

            self.mesh.Refine()
            ngmesh = self.mesh.ngmesh.Copy()
            self.mesh = ng.Mesh(ngmesh)
            self.mesh.Curve(8)
            if trustme:
                center = avr_zsqr
                npts = 1
                nspan = 1

        # Adaptivity loop done ------------------------------------------

        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        maxbdrnrm_r = np.max(self.boundarynorm(E_r))
        maxbdrnrm_l = np.max(self.boundarynorm(E_l))
        maxbdrnrm = max(maxbdrnrm_r, maxbdrnrm_l)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return Zsqrs, ndofs, E_r, E_l, phi_r, phi_l, beta, P

    def __get_phi_from_E(self, E_r, E_l, b, c, d, dinv, Y):
        """
        Get phi_r, phi_l from E_r, E_l
        """
        phi_r, phi_l = NGvecs(Y, E_r.m), NGvecs(Y, E_l.m)
        bE_r, cE_l = phi_r.zeroclone(), phi_l.zeroclone()

        bE_r._mv[:] = -b.mat * E_r._mv
        bE_r._mv[:] += d.harmonic_extension_trans * bE_r._mv
        phi_r._mv[:] = dinv * bE_r._mv
        phi_r._mv[:] += d.inner_solve * bE_r._mv
        phi_r._mv[:] += d.harmonic_extension * phi_r._mv

        cE_l._mv[:] = -c.mat.H * E_l._mv
        cE_l._mv[:] += d.harmonic_extension_trans.H * cE_l._mv
        phi_l._mv[:] = dinv.H * cE_l._mv
        phi_l._mv[:] += d.inner_solve.H * cE_l._mv
        phi_l._mv[:] += d.harmonic_extension.H * phi_l._mv

        return phi_r, phi_l

    def get_intensity(self, E, phi, betas):
        """
        Compute the intensity of the modes E = (E, -i/β phi L).
        Intensity = |E|^2 + |phi|^2 / |β|^2 L
        INPUTS:
        E: Transverse electric field
        phi: Longitudinal electric field
        zsqr: Non-dimensional Z² value of the mode
        OUTPUT:
        intensity: Intensity of the mode (as a list of CFs)
        """
        assert E.m == phi.m, f'E.m = {E.m} != phi.m = {phi.m}'
        assert len(betas) == E.m, f'len(betas) = {len(betas)} != E.m = {E.m}'
        intensity = []
        for i, beta in enumerate(betas):
            intensity.append(
                CF(
                    InnerProduct(E[i], E[i]) + InnerProduct(phi[i], phi[i]) /
                    (abs(beta)**2 * self.L)))
        return intensity

    # ###################################################################
    # GUIDED LP MODES FROM SELFADJOINT EIGENPROBLEM #####################

    def selfadjsystem(self, p):

        if self.ngspmlset:
            raise RuntimeError('NGSolve pml mesh trafo set.')
        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()
        A = ng.BilinearForm(X)
        A += grad(u) * grad(v) * dx + self.V * u * v * dx
        B = ng.BilinearForm(X)
        B += u * v * dx

        with ng.TaskManager():
            try:
                A.Assemble()
                B.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                B.Assemble()

        return A, B, X

    def selfadjmodes(self,
                     interval=(-10, 0),
                     p=3,
                     seed=1,
                     npts=20,
                     nspan=15,
                     within=None,
                     rhoinv=0.0,
                     quadrule='circ_trapez_shift',
                     verbose=True,
                     inverse='umfpack',
                     **feastkwargs):
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
        ctr = (right + left) / 2
        rad = (right - left) / 2
        P = SpectralProjNG(X,
                           a.mat,
                           b.mat,
                           radius=rad,
                           center=ctr,
                           npts=npts,
                           reduce_sym=True,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           inverse=inverse,
                           verbose=verbose)
        Y = NGvecs(X, nspan, M=b.mat)
        Y.setrandom(seed=seed)
        Zsqrs, Y, history, _ = P.feast(Y, hermitian=True, **feastkwargs)
        betas = self.betafrom(Zsqrs)

        return betas, Zsqrs, Y

    # ###################################################################
    # VECTOR MODES

    def vecmodesystem(self, p, alpha=None, inverse=None):
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
            radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'Outer')
            print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                  'and thickness=%.3f' % (self.Rout - self.R))
        elif self.ngspmlset:
            raise RuntimeError('Unexpected NGSolve pml mesh trafo here.')

        n = self.index
        n2 = n * n
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True)
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
        D = ng.BilinearForm(Y, condense=True)
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
            # Dinv = D.mat.Inverse(Y.FreeDofs(), inverse=inverse)
            Dinv = D.mat.Inverse(Y.FreeDofs(coupling=True), inverse=inverse)

        # resolvent of the vector mode problem --------------------------
        class ResolventVectorMode():

            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY)
            wrk2 = ng.GridFunction(XY)
            tmpY1 = ng.GridFunction(Y)
            tmpY2 = ng.GridFunction(Y)
            tmpX1 = ng.GridFunction(X)

            def __init__(selfr, z, V, n, inverse=None):
                n2 = n * n
                XY = ng.FESpace([X, Y])
                (E, phi), (v, psi) = XY.TnT()

                # selfr.zminusOp = ng.BilinearForm(XY)
                # selfr.zminusOp += (z * E * v - curl(E) * curl(v)
                #                    - V * E * v - grad(phi) * v
                #                    - n2 * phi * psi + n2 * E * grad(psi))
                # * dx
                # with ng.TaskManager():
                #     try:
                #         selfr.zminusOp.Assemble()
                #     except Exception:
                #         print('*** Trying again with larger heap')
                #         ng.SetHeapSize(int(1e9))
                #         selfr.zminusOp.Assemble()
                #     selfr.R = selfr.zminusOp.mat.Inverse(XY.FreeDofs(),
                #                                          inverse=inverse)

                selfr.Z = ng.BilinearForm(XY, condense=True)
                selfr.Z += (z * E * v - curl(E) * curl(v) - V * E * v -
                            grad(phi) * v - n2 * phi * psi +
                            n2 * E * grad(psi)) * dx
                selfr.ZH = ng.BilinearForm(XY, condense=True)
                selfr.ZH += (np.conjugate(z) * E * v - curl(E) * curl(v) -
                             V * E * v - grad(psi) * E - n2 * phi * psi +
                             n2 * v * grad(phi)) * dx
                with ng.TaskManager():
                    try:
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    selfr.R_I = selfr.Z.mat.Inverse(XY.FreeDofs(coupling=True),
                                                    inverse=inverse)

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

                        # selfr.wrk2.vec.data = selfr.R * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.Z.harmonic_extension_trans * selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.harmonic_extension * selfr.wrk2.vec

                        Rv._mv[i][:] = selfr.wrk2.components[0].vec

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

                        # selfr.wrk2.vec.data = selfr.R.H * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.ZH.harmonic_extension_trans * selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I.H * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.harmonic_extension * selfr.wrk2.vec

                        RHv._mv[i][:] = selfr.wrk2.components[0].vec

            def rayleigh_nsa(selfr,
                             ql,
                             qr,
                             qAq=not None,
                             qBq=not None,
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
                            # selfr.tmpY2.vec.data = Dinv * selfr.tmpY1.vec

                            selfr.tmpY1.vec.data += \
                                D.harmonic_extension_trans * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data = Dinv * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                D.inner_solve * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                D.harmonic_extension * selfr.tmpY2.vec

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

        return ResolventVectorMode, M.mat, A.mat, B.mat, C.mat, D, Dinv

    def guidedvecmodes(self,
                       rad,
                       ctr,
                       p=3,
                       seed=None,
                       npts=8,
                       nspan=20,
                       within=None,
                       rhoinv=0.0,
                       quadrule='circ_trapez_shift',
                       verbose=True,
                       inverse='umfpack',
                       **feastkwargs):
        """
        Capture guided vector modes whose non-dimensional resonance value Z²
        is such that Z*Z is within the interval (ctr-rad, ctr+rad).
        """

        R, M, A, B, C, D, Dinv = self.vecmodesystem(p, inverse=inverse)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        E.setrandom(seed=seed)

        print('Using FEAST to search for vector guided modes in')
        print(f'circle of radius {rad} centered at {ctr}')
        print(f'assuming not more than {nspan} modes in this interval.')
        print(f'System size: {E.n} x {E.n}  Inverse type: {inverse}')

        P = SpectralProjNGR(
            lambda z: R(z, self.V, self.index, inverse=inverse),
            radius=rad,
            center=ctr,
            npts=npts,
            within=within,
            rhoinv=rhoinv,
            quadrule=quadrule,
            verbose=verbose)
        Zsqrs, E, history, _ = P.feast(E, **feastkwargs)
        betas = self.betafrom(Zsqrs)

        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv

        BE._mv[:] += D.harmonic_extension_trans * BE._mv
        phi._mv[:] = Dinv * BE._mv
        phi._mv[:] += D.inner_solve * BE._mv
        phi._mv[:] += D.harmonic_extension * phi._mv

        return betas, Zsqrs, E, phi, R

    def leakyvecmodes(self,
                      rad,
                      ctr,
                      alpha=1,
                      p=3,
                      seed=1,
                      npts=8,
                      nspan=20,
                      within=None,
                      rhoinv=0.0,
                      quadrule='circ_trapez_shift',
                      verbose=True,
                      inverse='umfpack',
                      **feastkwargs):
        """
        Capture leaky vector modes whose non-dimensional resonance value Z²
        is contained  within the circular contour centered at "ctr"
        of radius "rad" in the Z² complex plane (not the Z-plane!).
        """

        R, M, A, B, C, D, Dinv = self.vecmodesystem(p,
                                                    alpha=alpha,
                                                    inverse=inverse)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        El = E.create()
        E.setrandom(seed=seed)
        El.setrandom(seed=seed)

        print('Using FEAST to search for vector leaky modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval' % nspan)
        print('System size:', E.n, ' x ', E.n, '  Inverse type:', inverse)

        P = SpectralProjNGR(
            lambda z: R(z, self.V, self.index, inverse=inverse),
            radius=rad,
            center=ctr,
            npts=npts,
            within=within,
            rhoinv=rhoinv,
            quadrule=quadrule,
            verbose=verbose)

        Zsqrs, E, history, El = P.feast(E,
                                        Yl=El,
                                        hermitian=False,
                                        **feastkwargs)
        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv

        BE._mv[:] += D.harmonic_extension_trans * BE._mv
        phi._mv[:] = Dinv * BE._mv
        phi._mv[:] += D.inner_solve * BE._mv
        phi._mv[:] += D.harmonic_extension * phi._mv

        betas = self.betafrom(Zsqrs)
        print('Results:\n Z²:', Zsqrs)
        print(' beta:', betas)
        print(' CL dB/m:', 20 * betas.imag / np.log(10))

        return betas, Zsqrs, E, phi, R

    # ###################################################################
    # BENT MODES

    def bentscalarmodes(self,
                        rad,
                        ctr,
                        R_bend,
                        p=2,
                        alpha=None,
                        npts=6,
                        nspan=10,
                        within=None,
                        rhoinv=0.0,
                        niterations=10,
                        quadrule='circ_trapez_shift',
                        verbose=True,
                        inverse='umfpack',
                        nrestarts=0,
                        **feastkwargs):
        """Find bent modes using scalar method.

        Bending radius R_bend should be non-dimensional. If alpha is provided,
        radial pml from ngsolve is set."""

        if alpha is not None:
            self.ngspmlset = True
            radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'Outer')
            print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                  'and thickness=%.3f' % (self.Rout - self.R))

        r = ng.x + R_bend
        k = self.k * self.index * self.L

        X = ng.H1(self.mesh, order=p, complex=True)

        u, v = X.TnT()

        A0 = ng.BilinearForm(X, check_unused=False)
        A0 += -r * ng.grad(u) * ng.grad(v) * ng.dx
        A0 += k**2 * r * u * v * ng.dx

        A1 = ng.BilinearForm(X, check_unused=False)
        A1 += R_bend**2 / r * u * v * ng.dx

        AA = [A0, A1]

        with ng.TaskManager():
            for i in range(len(AA)):
                try:
                    AA[i].Assemble()
                except Exception:
                    print('*** Trying again with larger heap')
                    ng.SetHeapSize(int(1e9))
                    AA[i].Assemble()

        P = SpectralProjNG(X,
                           A0.mat,
                           A1.mat,
                           radius=rad,
                           center=ctr,
                           npts=npts,
                           rhoinv=rhoinv,
                           quadrule=quadrule)

        Y = NGvecs(X, nspan)
        Yl = Y.create()
        Y.setrandom(seed=1)
        Yl.setrandom(seed=1)

        print('Using FEAST to search for scalar bent modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval.' % nspan)
        print('System size:', X.ndof, ' x ', X.ndof)
        print('  Inverse type:', inverse)

        Nu_sqrs, Y, hist, _ = P.feast(Y,
                                      Yl,
                                      hermitian=False,
                                      nrestarts=nrestarts,
                                      niterations=niterations,
                                      **feastkwargs)

        nus = (Nu_sqrs**.5) * R_bend
        CLs = 20 * nus.imag / np.log(10)
        print('Results:\n Nu²:', Nu_sqrs)
        print(' Nus:', nus)
        print(' CL dB/m:', CLs)

        return Nu_sqrs, Y, P, hist, CLs

    def bentmodesystem(self, p, R_bend, alpha=None, inverse=None):
        """
        Prepare eigensystem and resolvents for solving for bent vector modes.

        INPUTS:

        p: Determines degree of Nedelec x Lagrange space system.
           This should be an integer >= 0.

        R_bend: Non-dimensionalized bend radius.  Typical values around
            800 * R_clad.

        alpha: If alpha is None, prepare system for vector guided modes.
           If alpha is a positive number, use it as PML strength and
           prepare system for leaky modes using NGSolve's automatic
           mesh-based PML.
        """

        if alpha is not None:
            self.ngspmlset = True
            radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'Outer')
            print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                  'and thickness=%.3f' % (self.Rout - self.R))

        n = self.index
        n2 = n * n
        r = ng.x + R_bend  # r is NOT sqrt(x^2 + y^2)
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True)
        E, v = X.TnT()
        phi, psi = Y.TnT()

        A = ng.BilinearForm(X)
        A += r * (curl(E) * curl(v) - (self.L * self.k)**2 * n2 * E * v) * dx
        M = ng.BilinearForm(X)
        M += -R_bend**2 / r * E * v * dx
        C = ng.BilinearForm(trialspace=Y, testspace=X)
        C += R_bend * grad(phi) * v * dx
        C += R_bend / r * phi * v[0] * dx
        B = ng.BilinearForm(trialspace=X, testspace=Y)
        B += -n2 * r * E * grad(psi) * dx
        D = ng.BilinearForm(Y)
        # D = ng.BilinearForm(Y, condense=True)
        D += n2 * R_bend * phi * psi * dx

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
            Dinv = D.mat.Inverse(Y.FreeDofs(), inverse=inverse)
            # Dinv = D.mat.Inverse(Y.FreeDofs(coupling=True), inverse=inverse)
        # resolvent of the vector mode problem --------------------------

        class ResolventVectorMode():

            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY)
            wrk2 = ng.GridFunction(XY)
            tmpY1 = ng.GridFunction(Y)
            tmpY2 = ng.GridFunction(Y)
            tmpX1 = ng.GridFunction(X)

            def __init__(selfr, z, n, inverse=None):
                n2 = n * n
                XY = ng.FESpace([X, Y])
                (E, phi), (v, psi) = XY.TnT()
                selfr.zminusOp = ng.BilinearForm(XY)
                selfr.zminusOp += (
                    -z * R_bend**2 / r * E * v - r * curl(E) * curl(v) +
                    (self.L * self.k)**2 * n2 * r * E * v - R_bend *
                    (grad(phi) * v + 1 / r * phi * v[0]) -
                    R_bend * n2 * phi * psi + n2 * r * E * grad(psi)) * dx
                with ng.TaskManager():
                    try:
                        selfr.zminusOp.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.zminusOp.Assemble()
                    selfr.R = selfr.zminusOp.mat.Inverse(XY.FreeDofs(),
                                                         inverse=inverse)

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

            def rayleigh_nsa(selfr,
                             ql,
                             qr,
                             qAq=not None,
                             qBq=not None,
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

    def bentvecmodes(self,
                     rad,
                     ctr,
                     R_bend,
                     p=2,
                     alpha=None,
                     seed=1,
                     npts=6,
                     nspan=10,
                     rhoinv=0.0,
                     niterations=15,
                     nrestarts=0,
                     quadrule='circ_trapez_shift',
                     inverse='umfpack',
                     **feastkwargs):
        """
        Capture bent vector modes whose scaled propagation constants have real
        part near or in the interval L^2k_0^2 [n_clad^2, n_core^2].
        """

        R, M, A, B, C, D, Dinv = self.bentmodesystem(p,
                                                     R_bend,
                                                     alpha=alpha,
                                                     inverse=inverse)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        E.setrandom(seed=seed)

        print('Using FEAST to search for vector bent modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval.' % nspan)
        print('System size:', E.n, ' x ', E.n, '  Inverse type:', inverse)

        P = SpectralProjNGR(lambda z: R(z, self.index, inverse=inverse),
                            radius=rad,
                            center=ctr,
                            npts=npts,
                            rhoinv=rhoinv,
                            quadrule=quadrule)

        Nu_sqrs, E, history, _ = P.feast(E,
                                         hermitian=False,
                                         niterations=niterations,
                                         nrestarts=nrestarts,
                                         **feastkwargs)

        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv
        phi._mv[:] = Dinv * BE._mv

        nus = (Nu_sqrs**.5) * R_bend
        CLs = 20 * nus.imag / np.log(10)
        print('Results:\n Nu²:', Nu_sqrs)
        print(' Nus:', nus)
        print(' CL dB/m:', CLs)

        return Nu_sqrs, E, phi, R, CLs
