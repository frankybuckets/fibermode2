"""Facilities to NUMERICALLY compute transverse modes of a STEP-INDEX
fiber using a nondimensional eigenproblem and FEAST. Guided modes,
leaky modes, and bent modes can be computed.
"""

import ngsolve as ng
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import dx, BilinearForm, H1, CoefficientFunction, grad, IfPos
from fiberamp.fiber import Fiber
import fiberamp
from pyeigfeast.spectralproj.ngs import SpectralProjNG, NGvecs
from pyeigfeast.spectralproj.ngs import SpectralProjNGGeneral
from pyeigfeast.spectralproj import splitzoom
import sympy as sm
import os
from scipy.sparse import coo_matrix


class FiberMode:

    """A class to compute (guided and leaky) modes of a fiber in a
    nondimensional way. In nondimensional computations the core is
    set to have radius one. """

    def __init__(self, fibername=None, fromfile=None,
                 Rpml=None, Rout=None, geom=None,
                 h=4, hcore=None):
        """
        EITHER provide a prefix "filename" of a collection of files, e.g.,

            FiberMode(fromfile="filename")

        to reconstruct a previously saved object (ignoring other arguments),

        OR construct a new fiber geometry and mesh so that

          * region r < 1, in polar coords, is called "core",
          * region 1 < r < Rpml   is called "clad",
          * region Rpml < r < Rout   is called "pml",
          * when "Rpml" is unspecified, it is set to Rpml = (Rout+1)/2,
          * index of refraction is set using Fiber("fibername")
          * when "Rout" is unspecified, it is taken to match the ratio
            of cladding radius to core radius from Fiber("fibername"),
          * cladding and pml meshsize is "h", while core mesh size
            is "hcore" (set to a default of hcore = h/10),
          * degree "p" finite element space is set on the mesh.

        (Variables beginning with capital R such as "Rpml", "Rout" are
        nondimensional lengths -- in contrast, "rout" found in other classes
        is length in meters.)
        """

        self.outfolder = os.path.abspath(fiberamp.__path__[0]+'/outputs/')

        if fromfile is None:

            if fibername is None:
                raise ValueError('Need either a file or a fiber name')

            self.fibername = fibername
            self.fiber = Fiber(fibername)

            if Rout is None:
                Rout = self.fiber.rclad / self.fiber.rcore
            if Rpml is None:
                Rpml = (Rout+1)/2
            if Rpml < 1 or Rpml > Rout:
                raise ValueError('Set Rpml between 1 and Rout')
            self.Rpml = Rpml
            self.Rout = Rout

            if hcore is None:
                hcore = h/10
            self.hcore = hcore
            self.hclad = h
            self.hpml = h

            self.setstepindexgeom()  # sets self.geo
            mesh = ng.Mesh(self.geo.GenerateMesh())
            mesh.Curve(3)
            ng.Draw(mesh)
            self.mesh = mesh

        else:

            fbmfilename = self.outfolder+'/'+fromfile+'_fbm.npz'
            print('Loading FiberMode object from file ', fbmfilename)
            f = np.load(fbmfilename)
            self.fibername = str(f['fibername'])
            self.hcore = float(f['hcore'])
            self.hclad = float(f['hclad'])
            self.hpml = float(f['hpml'])
            self.Rpml = float(f['Rpml'])
            self.Rout = float(f['Rout'])

            self.fiber = Fiber(self.fibername)
            self.setstepindexgeom()  # sets self.geo

            meshfname = self.outfolder+'/'+fromfile+'_msh.vol.gz'
            print('Loading mesh from file ', meshfname)
            self.mesh = ng.Mesh(meshfname)
            self.mesh.ngmesh.SetGeometry(self.geo)
            self.mesh.Curve(3)
            ng.Draw(self.mesh)

        self.p = None        # degree of finite elements used in mode calc
        self.a = None
        self.b = None
        self.m = None
        self.pml_ngs = None  # True if ngsolve pml set (then cant reuse mesh)
        self.X = None

    # FURTHER INITIALIZATIONS & SETTERS #####################################

    def setstepindexgeom(self):

        geo = SplineGeometry()
        geo.AddCircle((0, 0), r=self.Rout,
                      leftdomain=1, rightdomain=0, bc='outer')
        geo.AddCircle((0, 0), r=self.Rpml,
                      leftdomain=2, rightdomain=1, bc='cladbdry')
        geo.AddCircle((0, 0), r=1,
                      leftdomain=3, rightdomain=2, bc='corebdry')
        geo.SetMaterial(1, 'pml')
        geo.SetMaterial(2, 'clad')
        geo.SetMaterial(3, 'core')

        geo.SetDomainMaxH(1, self.hpml)
        geo.SetDomainMaxH(2, self.hclad)
        geo.SetDomainMaxH(3, self.hcore)

        self.geo = geo

    def setrefractiveindex(self, curvature=12, bendfactor=1.28):
        """
        When a fiber of refractive index n is bent to have the
        input "curvature" (curvature = reciprocal of bending radius,
        since we assume bending along a perfect circle), the changed
        refratcive index is modeled by the formula

            nbent = n * (1 + (x * curvature/bendfactor))

        with "bendfactor" as input. This dimensional formula is used
        non-dimensionally below to set the internal data member "m",
        the non-dimensional coefficient function for the eigenproblem.
        """

        self.curvature = curvature
        self.bendfactor = bendfactor
        fib = self.fiber

        if curvature == 0:
            V = fib.fiberV()
            self.m = CoefficientFunction([0, 0, V*V])
        else:
            n = CoefficientFunction([fib.nclad, fib.nclad, fib.ncore])
            a = fib.rcore
            ka2 = (fib.ks * a) ** 2
            kan2 = ka2 * (fib.nclad ** 2)

            nbent = n * (1 + (ng.x * a * curvature/bendfactor))

            m = ka2 * nbent * nbent - kan2
            self.m = CoefficientFunction([0, m, m])

    # MODE CALCULATORS AND RELATED FUNCTIONALITIES  #########################

    def Z2toX2(self, Z2):
        """Convert non-dimensional Z² values to non-dimensional X² values
        through the relation X² - Z² = V². """

        Zsqr = np.array(Z2)
        Vsqr = self.fiber.fiberV()**2
        return Zsqr + Vsqr

    def X2toBeta(self, X2):
        """Convert non-dimensional X² values to dimensional propagation
        constants beta through the relation (ncore*k)² - X² = beta². """

        Xsqr = np.array(X2)
        a = self.fiber.rcore
        return np.sqrt((self.fiber.ks*self.fiber.ncore)**2 - Xsqr/a**2)

    def Z2toBeta(self, Z2):
        """Convert nondimensional Z² (input as "Z2") in the complex plane to
        complex propagation constant Beta. """

        return self.X2toBeta(self.Z2toX2(Z2))

    def ZtoBeta(self, Z):
        """Convert nondimensional Z in the complex plane to complex
        propagation constant Beta. """

        Z = np.array(Z)
        return np.sqrt((self.fiber.ks*self.fiber.ncore)**2
                       - (Z/self.fiber.rcore)**2)

    def guidedmodes(self, interval=None, p=3, nquadpts=20,
                    nspan=15, stop_tol=1e-10, check_contour=2,
                    niterations=50, verbose=True):
        """
        Search for guided modes in a given "interval" - which is to be
        input as a tuple: interval=(left, right).  If interval is None,
        then an automatic choice will be made to include all guided modes.

        The computation is done using Lagrangre finite elements of degree "p",
        with no PML, using selfadjoint FEAST with a random span of "nspan"
        vectors (and using the remaining parameters, which are simply
        passed to feast).

        OUTPUTS:

        betas, Zsqrs, Y: betas[i] give the i-th real-valued propagation
        constant and Zsqrs[i] gives the feast-computed i-th nondimensional
        Z² value in "interval". The corresponding eigenmode is i-th component
        of the span object Y.
        """

        self.p = p
        self.X = H1(self.mesh, order=self.p, dirichlet='outer', complex=True)

        if self.m is None:
            self.curvature = 0
            V = self.fiber.fiberV()
            self.m = CoefficientFunction([0, 0, V*V])
        if self.pml_ngs is True:
            raise RuntimeError('Mesh pml trafo is set')

        V = self.fiber.fiberV()
        if interval is None:

            # We choose the interval for the nondimensional Z² variable
            # recalling that  (a k₀ nclad)² < (β a)² < (a k₀ ncore)²,
            # where a is any scaling factor - and here it is rcore.
            # It follows that Z² = (a α₀)² = (a k₀ nclad)² - (a β)²
            # satisfies 0 > Z² > (a k₀ nclad)² - (a k₀ ncore)² = -V².

            left = -V*V
            right = 0
        else:
            left, right = interval

        u, v = self.X.TnT()

        a = BilinearForm(self.X)
        a += (grad(u) * grad(v) - self.m * u * v) * dx

        b = BilinearForm(self.X)
        b += u * v * dx

        with ng.TaskManager():
            a.Assemble()
            b.Assemble()
        self.a = a
        self.b = b

        print('Running selfadjoint FEAST to capture guided modes in (%g,%g)'
              % (left, right))
        print('assuming not more than %d modes in this interval' % nspan)

        ctr = (right+left)/2
        rad = (right-left)/2
        P = SpectralProjNG(self.X, a.mat, b.mat, rad, ctr, nquadpts,
                           reduce_sym=True, verbose=verbose)
        Y = NGvecs(self.X, nspan)
        Y.setrandom()
        Zsqrs, Y, history, _ = P.feast(Y, stop_tol=stop_tol,
                                       check_contour=check_contour,
                                       niterations=niterations)
        betas = np.array(self.Z2toBeta(Zsqrs))

        return betas, Zsqrs, Y

    def name2indices(self, betas, maxl=9, delta=None):
        """Given a numpy 1D array "betas" of approximations to
        propagation constants, produce a dictionary of mode names and
        corresponding exact propagation constants.

        OUTPUT of name2ind, exact = name2indices(betas)

            * name2ind is a dictionary such that beta[name2ind['LP01']]
              gives the beta corresponding to LP01 mod, etc.

            * exact[i] = i-th exact propagation constant obtained
              semi-analytically, to which beta[i] is an approximation.

        OPTIONAL INPUTS:

            delta: consider numbers that differ by less than delta as
            approximations of a multiple eigenvalue.

            maxl: assume that betas correspond to LP(l,m) modes where l is
            less than maxl.
        """

        V = self.fiber.fiberV()
        lft = self.Z2toBeta(0)  # betas must be in (lft, rgt)
        rgt = self.Z2toBeta(-V*V)
        # roughly identify simple and multiple ew approximants
        sm, ml = splitzoom.simple_multiple_zoom(lft, rgt, betas, delta=delta)

        name2ind = {}
        exact = -np.ones_like(betas)

        # l=0 case should be simple eigenvalues:
        activesimple = np.arange(len(sm['index']))
        LP0 = self.fiber.XtoBeta(self.fiber.propagation_constants(0))
        b = betas[sm['index']]
        for m in range(len(LP0)):
            ind = np.argmin(abs(LP0[m]-b[activesimple]))
            i2beta = sm['index'][activesimple[ind]]
            name2ind['LP0' + str(m+1)] = i2beta
            exact[i2beta] = LP0[m]
            activesimple = np.delete(activesimple, [ind])
            if len(activesimple) == 0:
                break

        # l>0 cases should have multiplicity 2:
        activemultiple = np.arange(len(ml['index']))
        ctrs = np.array(ml['center'])
        for l in range(1, maxl):
            LPl = self.fiber.XtoBeta(self.fiber.propagation_constants(l))
            for m in range(len(LPl)):
                ind = np.argmin(abs(LPl[m]-ctrs[activemultiple]))
                i2beta_a = ml['index'][activemultiple[ind]][0]
                i2beta_b = ml['index'][activemultiple[ind]][1]
                name2ind['LP' + str(l) + str(m+1)+'_a'] = i2beta_a
                name2ind['LP' + str(l) + str(m+1)+'_b'] = i2beta_b
                exact[i2beta_a] = LPl[m]
                exact[i2beta_b] = LPl[m]
                activemultiple = np.delete(activemultiple, ind)
                if len(activemultiple) == 0:
                    return name2ind, exact
        return name2ind, exact

    # LEAKY MODES ###########################################################

    def leakymode_auto(self, p, radiusZ2, centerZ2,
                       alpha=1, includeclad=False,
                       stop_tol=1e-10, npts=10, niter=50, nspan=10,
                       verbose=True, inverse='umfpack'):
        """Compute leaky modes by solving a linear eigenproblem using
        the frequency-independent automatic PML mesh map of NGSolve
        and using non-selfadjoint FEAST.

        INPUTS:

        * p: degree of finite element to be used to compute modes.
        * radiusZ2, centerZ2:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * alpha:  PML strength
        * includeclad:
            If True, then cladding is included in the domain, so PML
            is set in 'pml' region only.
            If False, then PML is set in the union 'pml|clad'.
        * npts: number of quadrature points in the contour for FEAST.
        * niter: number of FEAST iterations before restart.
        * nspan: intial number of random vectors to start FEAST.

        OUTPUTS:

        * zsqr: computed resonance values Z²
        * Yl, Y: left and right eigenspans
        * P: spectral projector approximation
        """

        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')

        self.alpha = alpha
        self.pml_ngs = True

        if includeclad:
            radial = ng.pml.Radial(rad=self.Rpml,
                                   alpha=alpha*1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'pml')
            pmlbegin = self.Rpml
        else:
            radial = ng.pml.Radial(rad=1, alpha=alpha*1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'pml|clad')
            pmlbegin = 1

        if self.m is None:
            self.setrefractiveindex(curvature=0)

        print(' PML (automatic, frequency-independent) starts at r=', pmlbegin)
        print(' Degree p = ', p, ' Curvature =', self.curvature)

        self.p = p
        self.X = H1(self.mesh, order=self.p, dirichlet='outer', complex=True)

        u, v = self.X.TnT()
        a = BilinearForm(self.X)
        b = BilinearForm(self.X)
        a += (grad(u) * grad(v) - self.m * u * v) * dx
        b += u * v * dx
        with ng.TaskManager():
            a.Assemble()
            b.Assemble()
        self.a = a
        self.b = b

        P = SpectralProjNGGeneral(self.X, self.a.mat, self.b.mat,
                                  radiusZ2, centerZ2, npts,
                                  verbose=verbose, inverse=inverse)
        Y = NGvecs(self.X, nspan)
        Yl = Y.create()
        Y.setrandom()
        Yl.setrandom()
        zsqr, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                       stop_tol=stop_tol,
                                       check_contour=2,
                                       niterations=niter, nrestarts=1)
        return zsqr, Yl, Y, P

    def leakymode_smooth(self, p, radiusZ2, centerZ2,
                         alpha=1, pmlbegin=None, pmlend=None,
                         stop_tol=1e-10, npts=10, niter=50,
                         verbose=True, inverse='umfpack'):
        """Compute leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Then a non-selfadjoint FEAST is run on the system.

        INPUTS:

        * p: degree of finite element to be used to compute modes.
        * radiusZ2, centerZ2:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * alpha:  PML strength α
        * pmlbegin, pmlend:  starting radius of the PML and ending radius
            of the transitional PML region, respectively. (The subdomains
            'pml', 'clad' in the mesh are not used for this PML.)
        * npts: number of quadrature points in the contour for FEAST.
        * niter: number of FEAST iterations before restart.

        OUTPUTS:

        * zsqr: computed resonance values Z²
        * Yl, Y: left and right eigenspans
        * P: spectral projector approximation
        """

        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = 1
        else:
            if pmlbegin > self.Rout or pmlbegin < 1:
                raise ValueError('Select pmlbegin in interval [1, %g]'
                                 % self.Rout)
            self.pml_ngs = False
        if pmlend is None:
            pmlend = (self.Rout+pmlbegin) * 0.5

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
        g = IfPos(r-pmlbegin, g0, 1)
        tt = IfPos(r-pmlbegin, tt0, 1)

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
        A = CoefficientFunction((A00, A01,
                                 A01, A11), dims=(2, 2))
        self.pml_A = A
        self.pml_tt = tt

        # Make linear eigensystem
        if self.m is None:
            self.setrefractiveindex(curvature=0)
        self.p = p
        print(' PML (smooth, k-independent) starts at r=', pmlbegin)
        print(' Degree p = ', p, ' Curvature =', self.curvature)

        self.X = H1(self.mesh, order=self.p, dirichlet='outer', complex=True)
        u, v = self.X.TnT()
        a = BilinearForm(self.X)
        b = BilinearForm(self.X)
        a += (self.pml_A * grad(u) * grad(v) -
              self.m * self.pml_tt * u * v) * dx
        b += self.pml_tt * u * v * dx
        with ng.TaskManager():
            a.Assemble()
            b.Assemble()
        self.a = a
        self.b = b

        # Use spectral projector to find the resonance values squared
        P = SpectralProjNGGeneral(self.X, self.a.mat, self.b.mat,
                                  radiusZ2, centerZ2, npts=npts,
                                  verbose=verbose, inverse=inverse)
        Y = NGvecs(self.X, 10)
        Yl = NGvecs(self.X, 10)
        Y.setrandom()
        Yl.setrandom()
        zsqr, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                       stop_tol=stop_tol,
                                       check_contour=2,
                                       niterations=niter, nrestarts=1)
        return zsqr, Yl, Y, P

    def leakymode_poly(self, p, radius, center,
                       alpha=1, includeclad=False,
                       stop_tol=1e-10, npts=10, niter=50, nspan=10,
                       verbose=True, inverse='umfpack'):
        """Compute leaky modes by solving a nonlinear polynomial eigenproblem
        using a frequency-dependent PML formulated by [Nannen+Wess]:
           mapped_x = x * η(r) / r,                   where
           η(r) = R + (r - R) * (1 + 1j * α) / Z
        and R is the radius where PML starts (the variable pmlbegin below).
        (Note that Z takes the role of frequency, called ω in [Nannen+Wess].)
        The polynomial eigenproblem is converted to a larger linear
        eigenproblem which is solved using non-selfadjoint FEAST.

        INPUTS:

        * p: degree of finite element to be used to compute modes.
        * radius, center: Capture modes whose non-dimensional resonance
            value Z (not Z²) is contained within the circular contour
            centered at "center" of radius "radius" in the complex plane.
        * alpha: The quantity α in the above mapping formula (PML strength).
        * includeclad:
            If True, then cladding is included in the domain, so PML
            is set in 'pml' region only.
            If False, then PML is set in the union 'pml|clad'.
        * npts: number of quadrature points in the contour for FEAST.
        * niter: number of FEAST iterations before restart.

        OUTPUTS:

        * z: computed resonance values
        * yl, y: computed left and right eigenspans
        * P: spectral projector approximation
        """

        if self.m is None:
            self.setrefractiveindex(curvature=0)
        self.p = p
        print(' PML (poly, k-dependent), includeclad =', includeclad)
        print(' Degree p = ', p, ' Curvature =', self.curvature)

        self.X = H1(self.mesh, order=self.p, dirichlet='outer', complex=True)

        # Our implementation of [Nannen+Wess]'s frequency-dependent PML is
        # based on the idea to make a cubic eigenproblem using 3 copies of X:

        X3 = ng.FESpace([self.X, self.X, self.X])

        u0, u1, u2 = X3.TrialFunction()
        v0, v1, v2 = X3.TestFunction()
        u0x, u0y = grad(u0)
        u1x, u1y = grad(u1)
        u2x, u2y = grad(u2)
        v2x, v2y = grad(v2)

        if includeclad:
            pmlbegin = self.rclad
            dx_pml = dx(definedon=self.mesh.Materials('pml'))
            dx_int = dx(definedon=self.mesh.Materials('core|clad'))
        else:
            pmlbegin = 1
            dx_pml = dx(definedon=self.mesh.Materials('pml|clad'))
            dx_int = dx(definedon=self.mesh.Materials('core'))

        R = pmlbegin
        s = 1 + 1j * alpha
        x = ng.x
        y = ng.y
        r = ng.sqrt(x*x+y*y) + 0j

        A = BilinearForm(X3)
        B = BilinearForm(X3)

        A += u1 * v0 * dx
        A += u2 * v1 * dx

        A += (s*r/R) * grad(u0) * grad(v2) * dx_pml
        A += s * (r-R)/(R*r*r) * (x*u0x+y*u0y) * v2 * dx_pml
        A += s * (R-2*r)/r**3 * (x*u0x+y*u0y) * (x*v2x+y*v2y) * dx_pml
        A += -s**3 * (r-R)**2/(R*r) * u0 * v2 * dx_pml

        A += grad(u1) * grad(v2) * dx_int
        A += -self.m * u1 * v2 * dx_int
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
                           radius, center, npts,
                           verbose=verbose, inverse=inverse)
        Y = NGvecs(X3, nspan, M=B.mat)
        Yl = Y.create()
        Y.setrandom()
        Yl.setrandom()

        z, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False,
                                    stop_tol=stop_tol,
                                    check_contour=2,
                                    niterations=niter, nrestarts=1)
        Yg = Y.gridfun()
        Ylg = Y.gridfun()
        y = NGvecs(self.X, Y.m)
        yl = NGvecs(self.X, Y.m)
        for i in range(Y.m):
            y.data[i].data = Yg.components[0].vecs[i]
            yl.data[i].data = Ylg.components[0].vecs[i]

        return z, yl, y, P, Yl, Y

    # BENT MODES ############################################################

    def bentmode(self, curvature, radiusZ, centerZ, p,
                 method='poly',
                 bendfactor=1.28,
                 includeclad=False, pmlbegin=None, pmlend=None, alpha=1,
                 npts=10, niter=50, stop_tol=1.e-10):

        self.setrefractiveindex(curvature=curvature, bendfactor=bendfactor)
        radiusZ2 = radiusZ ** 2
        centerZ2 = centerZ ** 2

        if method == 'poly':

            z, Yl, Y, P = self.leakymode_poly(p, radiusZ, centerZ,
                                              stop_tol=stop_tol, npts=npts,
                                              niter=niter, alpha=alpha)
            z2 = z ** 2
            Y.draw()

        elif method == 'auto':

            z2, Yl, Y, P = self.leakymode_auto(p, radiusZ2, centerZ2,
                                               alpha=alpha,
                                               includeclad=False,
                                               stop_tol=stop_tol,
                                               npts=npts, niter=niter)
            Y.draw()

        elif method == 'smooth':

            z2, Yl, Y, P = self.leakymode_smooth(p, radiusZ2, centerZ2,
                                                 alpha=alpha,
                                                 pmlbegin=pmlbegin,
                                                 pmlend=pmlend,
                                                 stop_tol=stop_tol,
                                                 npts=npts, niter=niter)
            Y.draw()

        else:
            raise ValueError('Unknown method')

        print('Nonlinear eigenvalues in nondimensional Z-squared plane:\n', z2)
        print('Physical propagation constants:\n', self.Z2toBeta(z2))
        return z2, Y, P

    # CONVENIENCE & DEBUGGING ###############################################

    def scipymats(self):
        """ Return scipy versions of matrices FiberMode.a and FiberMode.b,
        if these data members exist. (Also uses FiberMode.X freedofs.)"""

        if self.a is None or self.b is None or self.X is None:
            raise RuntimeError('Set a, b, and X before calling scipymats()')

        free = np.array(self.X.FreeDofs())
        freedofs = np.where(free)[0]
        i, j, avalues = self.a.mat.COO()
        A = coo_matrix((avalues.NumPy(), (i, j)))
        i, j, bvalues = self.b.mat.COO()
        B = coo_matrix((bvalues.NumPy(), (i, j)))
        A = A.tocsc()[:, freedofs]
        A = A.tocsr()[freedofs, :]
        B = B.tocsc()[:, freedofs]
        B = B.tocsr()[freedofs, :]
        return A, B, freedofs

    # SAVING & LOADING ######################################################
    #
    # File naming conventions:
    #  * File output sets are classified by a prefix name <prefix>
    #  * FiberMode object saved in file: <prefix>_fbm.npz
    #  * Mesh saved in file:             <prefix>_msh.vol.gz
    #  * Modes saved in file(s):         <prefix>_mde.npz
    #

    def savefbm(self, fileprefix):
        """ Save this object so it can be loaded later """

        if os.path.isdir(self.outfolder) is not True:
            os.mkdir(self.outfolder)
        fbmfilename = self.outfolder+'/'+fileprefix+'_fbm.npz'
        print('Writing FiberMode object into:\n', fbmfilename)
        np.savez(fbmfilename,
                 fibername=self.fibername,
                 hcore=self.hcore, hclad=self.hclad, hpml=self.hpml,
                 Rpml=self.Rpml, Rout=self.Rout)

    def savemesh(self, fileprefix):

        meshfname = self.outfolder+'/'+fileprefix+'_msh.vol.gz'
        print('Writing mesh into:\n', meshfname)
        self.mesh.ngmesh.Save(meshfname)

    def savemodes(self, fileprefix, betas, Y,
                  saveallagain=True, name2ind=None, exact=None):
        """ Convert Y to numpy and save in npz format. """

        if saveallagain:
            self.savefbm(fileprefix)
            self.savemesh(fileprefix)

        y = Y.tonumpy()
        if os.path.isdir(self.outfolder) is not True:
            os.mkdir(self.outfolder)
        fullname = self.outfolder+'/'+fileprefix+'_mde.npz'
        print('Writing modes into:\n', fullname)
        np.savez(fullname, fibername=self.fibername,
                 hcore=self.hcore, hclad=self.hclad, hpml=self.hpml,
                 p=self.p, Rpml=self.Rpml, Rout=self.Rout,
                 betas=betas, y=y,
                 exactbetas=exact, name2ind=name2ind)

    def checkload(self, f):
        """Check if the loaded file has expected values of certain data"""

        for member in {'fibername', 'hcore', 'hclad', 'hpml',
                       'Rpml', 'Rout'}:
            print('  From file:', member, '=', f[member])
            assert self.__dict__[member] == f[member], \
                'Load error! Data member %s does not match!' % member

    def loadmodes(self, modefile):
        """Load modes from "outputs/modefile" (filename with extension)"""

        fname = self.outfolder+'/'+modefile
        print('Loading modes from:\n ', fname)
        f = np.load(fname, allow_pickle=True)
        self.checkload(f)
        self.p = int(f['p'])
        print('  Degree %d modes found in file' % self.p)
        self.X = H1(self.mesh, order=self.p, dirichlet='outer', complex=True)
        y = f['y']
        betas = f['betas']
        n2i = f['name2ind'].item()
        m = y.shape[0]
        Y = NGvecs(self.X, m)
        Y.fromnumpy(y)
        return betas, Y, n2i

    def makeguidedmodelibrary(self, maxp=5,
                              maxl=9, delta=None, nspan=15):
        """Save full sets of guided modes computed using the same mesh, using
        polynomial degrees p from 1 to "maxp", together with their LP
        names. One modefile per p is written and all output filenames
        are prefixed with fiber's name. (Remaining optional arguments
        are passed to name2indices(..), where they are also documented.)
        """

        fprefix = self.fibername
        self.savefbm(fprefix)        # save FiberMode object
        self.savemesh(fprefix)       # save mesh

        for p in range(1, maxp+1):   # save modes, one file per degree
            betas, zsqrs, Y = self.guidedmodes(p=p, nspan=nspan)
            print('Physical propagation constants:\n', betas)
            print('Computed non-dimensional Z-squared values:\n', zsqrs)
            n2i, exbeta = self.name2indices(betas, maxl=maxl, delta=delta)
            self.savemodes(fprefix+'_p' + str(p), betas, Y,
                           saveallagain=False, name2ind=n2i, exact=exbeta)
