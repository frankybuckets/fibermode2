"""Facilities to NUMERICALLY compute transverse modes of a STEP-INDEX
fiber using a nondimensional eigenproblem and FEAST. Guided modes,
leaky modes, and bent modes can be computed.
"""

import ngsolve as ng
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import dx, BilinearForm, H1, CoefficientFunction, grad, IfPos
from fiberamp.fiber import Fiber
from pyeigfeast.spectralproj.ngs import SpectralProjNG, NGvecs
from pyeigfeast.spectralproj import splitzoom
import sympy as sm
import os
from cmath import exp, pi, phase, sqrt


class FiberMode:

    """A class to compute (guided and leaky) modes of a fiber in a
    nondimensional way. In nondimensional computations the core is
    set to have radius one. """

    def __init__(self, fibername=None, fromfile=None,
                 rpml=None, rout=None, geom=None,
                 h=4, hcore=None):
        """
        EITHER provide a prefix "filename" of a collection of files, e.g.,

            FiberMode(fromfile="filename")

        to reconstruct a previously saved object (ignoring other arguments),

        OR construct a new fiber geometry and mesh so that

          * region r < 1, in polar coords, is called "core",
          * region 1 < r < rpml   is called "clad",
          * region rpml < r < rout   is called "pml",
          * when "rpml" is unspecified, it is set to rpml = (rout+1)/2,
          * index of refraction is set using Fiber("fibername")
          * when "rout" is unspecified, it is taken to match the ratio
            of cladding radius to core radius from Fiber("fibername"),
          * cladding and pml meshsize is "h", while core mesh size
            is "hcore" (set to a default of hcore = h/10),
          * degree "p" finite element space is set on the mesh.
        """

        if fromfile is None:

            if fibername is None:
                raise ValueError('Need either a file or a fiber name')

            self.fibername = fibername
            self.fiber = Fiber(fibername)

            if rout is None:
                rout = self.fiber.rclad / self.fiber.rcore
            if rpml is None:
                rpml = (rout+1)/2
            if rpml < 1 or rpml > rout:
                raise ValueError('Set rpml between 1 and rout')
            self.rpml = rpml
            self.rout = rout

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

            fbmfilename = os.path.abspath('../../outputs/'+fromfile+'_fbm.npz')
            print('Loading FiberMode object from file ', fbmfilename)
            f = np.load(fbmfilename)
            self.fibername = str(f['fibername'])
            self.hcore = float(f['hcore'])
            self.hclad = float(f['hclad'])
            self.hpml = float(f['hpml'])
            self.rpml = float(f['rpml'])
            self.rout = float(f['rout'])

            self.fiber = Fiber(self.fibername)
            self.setstepindexgeom()  # sets self.geo

            meshfname = os.path.abspath(
                '../../outputs/'+fromfile+'_msh.vol.gz')
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
        geo.AddCircle((0, 0), r=self.rout,
                      leftdomain=1, rightdomain=0, bc='outer')
        geo.AddCircle((0, 0), r=self.rpml,
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

    def setrefractiveindex(self, curvature=12, curvefactor=1.28):
        """
        When a fiber of refractive index n is bent to have the
        input "curvature" (curvature = reciprocal of bending radius,
        since we assume bending along a perfect circle), the changed
        refratcive index is modeled by the formula

            nbent = n * (1 + (x * curvature/curvefactor))

        with "curvefactor" as input. This dimensional formula is used
        non-dimensionally below to set the internal data member "m",
        the non-dimensional coefficient function for the eigenproblem.
        """

        self.curvature = curvature
        self.curvefactor = curvefactor
        fib = self.fiber

        if curvature == 0:
            V = fib.fiberV()
            self.m = CoefficientFunction([0, 0, V*V])
        else:
            n = CoefficientFunction([fib.nclad, fib.nclad, fib.ncore])
            a = fib.rcore
            ka2 = (fib.ks * a) ** 2
            kan2 = ka2 * (fib.nclad ** 2)

            nbent = n * (1 + (ng.x * a * curvature/curvefactor))

            m = ka2 * nbent * nbent - kan2
            self.m = CoefficientFunction([0, m, m])

    def setpmlcoef(self, method, alpha=1,
                   includeclad=False,
                   pmlbegin=None, pmlend=None):
        """
        Make linear systems for frequency-independent PML, using
        input alpha = PML strength (of exponential decay).

        When method=auto, pmlbegin and pmlend are ignored, while
        when method=smooth, includeclad is ignored.

        Check docstring of leakymode method for documentation of options.
        """

        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        self.alpha = alpha

        if method == 'auto':

            self.pml_ngs = True
            if includeclad:
                radial = ng.pml.Radial(rad=self.rpml,
                                       alpha=alpha*1j, origin=(0, 0))
                self.mesh.SetPML(radial, 'pml')
                pmlbegin = self.rpml
            else:
                radial = ng.pml.Radial(rad=1, alpha=alpha*1j, origin=(0, 0))
                self.mesh.SetPML(radial, 'pml|clad')
                pmlbegin = 1
            print(' PML (automatic, k-independent) starts at r=', pmlbegin)

        elif method == 'smooth':

            if pmlbegin is None:
                pmlbegin = self.rpml
            else:
                if pmlbegin > self.rout or pmlbegin < 1:
                    raise ValueError('pmlbegin should be between 1 and %g' %
                                     self.rout)
            self.pml_ngs = False

            if pmlend is None:
                pmlend = (self.rout+pmlbegin) * 0.5

            # symbolically derive the radial PML functions
            s, t, R0, R1 = sm.symbols('s t R_0 R_1')
            nr = sm.integrate((s-R0)**2 * (s-R1)**2, (s, R0, t)).factor()
            dr = nr.subs(t, R1).factor()
            sigmat = alpha * nr / dr
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

            print(' PML (smooth, k-independent) starts at r=', pmlbegin)

        elif method == 'poly' or method == 'poleff':

            raise ValueError('Dont use setpmlcoef(..) for these methods')

        else:

            raise NotImplementedError()

    # FUNCTIONALITIES  ####################################################

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
        """
        Convert nondimensional Z² (input as "Z2") in the complex plane
        to complex propagation constant Beta.
        """
        return self.X2toBeta(self.Z2toX2(Z2))

    def sqrt(self, c):
        """
        Return sqrt(c) taking branch cut along arg(z) = pi + pi/8 (not the
        usual negative real axis arg(z) = pi).
        """
        p = phase(c*exp(-1j*pi/8)) + pi/8
        return sqrt(abs(c)) * exp(1j*p/2)

    def ZtoBeta(self, Z):
        """
        Convert nondimensional Z in the complex plane to complex propagation
        constant Beta.
        """
        return self.sqrt((self.fiber.ks*self.fiber.ncore)**2
                         - (Z/self.fiber.rcore)**2)

    def guidedmodes(self, interval=None, p=3, nquadpts=20,
                    numvecs=25, stop_tol=1e-10, check_contour=2,
                    niterations=50, verbose=True):
        """
        Search for guided modes in a given "interval" - which is to be
        input as a tuple: interval=(left, right).  If interval is None,
        then an automatic choice will be made to include all guided modes.

        The computation is done using Lagrangre finite elements of degree "p",
        with no PML, using selfadjoint FEAST with a random span of "numvecs"
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

        print('Running selfadjoint FEAST to capture guided modes in (%g,%g)'
              % (left, right))
        print('assuming not more than %d modes in this interval' % numvecs)

        ctr = (right+left)/2
        rad = (right-left)/2
        P = SpectralProjNG(self.X, a.mat, b.mat, rad, ctr, nquadpts,
                           reduce_sym=True, verbose=verbose)
        Y = NGvecs(self.X, numvecs)
        Y.setrandom()
        Y.draw()
        Zsqrs, Y, history, _ = P.feast(Y, stop_tol=stop_tol,
                                       check_contour=check_contour,
                                       niterations=niterations)
        betas = np.array(self.Z2toBeta(Zsqrs))

        return betas, Zsqrs, Y

    def name2indices(self, betas, maxl=9, delta=None):
        """Given numpy 1D array "betas" of approximations to propagation
        constants, produce a dictionary of mode names and errors in
        propagation constants.

        OUTPUT of name2ind, exact = name2indices(betas)

            * name2ind is a dictionary such that beta[name2ind['LP01']]
              gives the beta corresponding to LP01 mod, etc.

            * exact[i] = i-th exact propagation constant obtained
              semi-analytically, to which beta[i] is an approximation.

        OPTIONAL INPUTS:

            delta: consider numbers that differ by less than delta as
            approximations of a multiple eigenvalue.

            maxl: assume that betas consider LP(l,m) modes where l is
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

        # l>0 cases should have multiplicity 2:
        activemultiple = np.arange(len(ml['index']))
        ctrs = np.array(ml['center'])
        for l in range(1, maxl):
            LPl = self.fiber.XtoBeta(self.fiber.propagation_constants(l))
            for m in range(len(LPl)):
                ind = np.argmin(abs(LPl[m]-ctrs))
                i2beta_a = ml['index'][activemultiple[ind]][0]
                i2beta_b = ml['index'][activemultiple[ind]][1]
                name2ind['LP' + str(l) + str(m+1)+'_a'] = i2beta_a
                name2ind['LP' + str(l) + str(m+1)+'_b'] = i2beta_b
                exact[i2beta_a] = LPl[m]
                exact[i2beta_b] = LPl[m]
                activemultiple = np.delete(activemultiple,
                                           [i2beta_a, i2beta_b])

        return name2ind, exact

    # SAVING & LOADING ######################################################

    def savefbm(self, fileprefix):
        """ Save this object so it can be loaded later """

        if os.path.isdir('../../outputs') is not True:
            os.mkdir('../../outputs')
        fbmfilename = os.path.abspath('../../outputs/'+fileprefix+'_fbm.npz')
        print('Writing FiberMode object into:\n', fbmfilename)
        np.savez(fbmfilename,
                 fibername=self.fibername,
                 hcore=self.hcore, hclad=self.hclad, hpml=self.hpml,
                 rpml=self.rpml, rout=self.rout)

    def savemesh(self, fileprefix):

        meshfname = os.path.abspath('../../outputs/'+fileprefix+'_msh.vol.gz')
        print('Writing mesh into:\n', meshfname)
        self.mesh.ngmesh.Save(meshfname)

    def savemodes(self, fileprefix, betas, Y,
                  saveallagain=True, name2ind=None, exact=None):
        """ Convert Y to numpy and save in npz format. """

        if saveallagain:
            self.savefbm(fileprefix)
            self.savemesh(fileprefix)

        y = Y.tonumpy()
        if os.path.isdir('../../outputs') is not True:
            os.mkdir('../../outputs')
        fullname = os.path.abspath('../../outputs/'+fileprefix+'_mde.npz')
        print('Writing modes into:\n', fullname)
        np.savez(fullname, fibername=self.fibername,
                 hcore=self.hcore, hclad=self.hclad, hpml=self.hpml,
                 p=self.p, rpml=self.rpml, rout=self.rout,
                 betas=betas, y=y,
                 exactbetas=exact, name2ind=name2ind)

    def checkload(self, f):
        """Check if the loaded file has expected values of certain data"""

        for member in {'fibername', 'hcore', 'hclad', 'hpml',
                       'rpml', 'rout'}:
            print('  From file:', member, '=', f[member])
            assert self.__dict__[member] == f[member], \
                'Load error! Data member %s does not match!' % member

    def loadmodes(self, modefile):
        """Load modes from "outputs/modefile" (filename with extension)"""

        fname = os.path.abspath('../../outputs/'+modefile)
        print('Loading modes from:\n ', fname)
        f = np.load(fname)
        self.checkload(f)
        self.p = int(f['p'])
        print('  Degree %d modes found in file' % self.p)
        self.X = H1(self.mesh, order=self.p, dirichlet='outer', complex=True)
        y = f['y']
        betas = f['betas']
        m = y.shape[0]
        Y = NGvecs(self.X, m)
        Y.fromnumpy(y)
        return betas, Y

    def makeguidedmodelibrary(self, maxp=5,
                              maxl=9, delta=None):
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
            betas, zsqrs, Y = self.guidedmodes(p=p)
            print('Physical propagation constants:\n', betas)
            print('Computed non-dimensional Z-squared values:\n', zsqrs)
            n2i, exbeta = self.name2indices(betas, maxl=maxl, delta=delta)
            self.savemodes(fprefix+'_p' + str(p), betas, Y,
                           saveallagain=False, name2ind=n2i, exact=exbeta)
