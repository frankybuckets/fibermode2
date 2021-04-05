import ngsolve as ng
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import H1, CoefficientFunction, IfPos
from ngsolve.special_functions import jv, kv
from fiberamp.fiber import Fiber
import fiberamp
from pyeigfeast.spectralproj.ngs import NGvecs
from pyeigfeast.spectralproj import splitzoom
from fiberamp.fiber.modesolver import ModeSolver
import os
from scipy.sparse import coo_matrix
import scipy.special as scf


class FiberMode(ModeSolver):

    """Class with facilities to numerically approximate transverse modes
    of a STEP-INDEX fiber using a nondimensional eigenproblem and FEAST.
    Guided modes and leaky modes can be computed.
    """

    def __init__(self, fibername=None, fromfile=None,
                 R=None, Rout=None, geom=None,
                 h=3, hcore=None, refine=0):
        """
        EITHER provide a prefix "filename" of a collection of files, e.g.,

            FiberMode(fromfile="filename")

        to reconstruct a previously saved object (ignoring other arguments),

        OR construct a new fiber geometry and mesh so that

          * region r < 1, in polar coords, is called "core",
          * region 1 < r < R   is called "clad",
          * region R < r < Rout   is called "pml",
          * when "R" is None, it is set to R = (Rout+1)/2,
          * index of refraction is set using Fiber("fibername")
          * when "Rout" is unspecified, it is taken to match the ratio
            of cladding radius to core radius from Fiber("fibername"),
          * cladding and pml meshsize is "h", while core mesh size
            is "hcore" (set to a default of hcore = h/10),
          * degree "p" finite element space is set on the mesh.

        (Variables beginning with capital R such as "R", "Rout" are
        nondimensional lengths -- in contrast, "rout" found in other classes
        is length in meters.)
        """

        self.outfolder = os.path.abspath(fiberamp.__path__[0]+'/outputs/')

        if fromfile is None:
            if fibername is None:
                raise ValueError('Need either a file or a fiber name')
            self.makefibermode(fibername, R=R, Rout=Rout, geom=geom,
                               h=h, hcore=hcore)
            self.makemesh(refine)
        else:
            fbmfilename = self.outfolder+'/'+fromfile+'_fbm.npz'
            if os.path.isfile(fbmfilename):
                self.loadfibermode(fbmfilename)
            else:
                print('Specified fibermode file not found -- creating it')
                self.makefibermode(fromfile)
                self.savefbm(fromfile)

            meshfname = self.outfolder+'/'+fromfile+'_msh.vol.gz'
            if os.path.isfile(meshfname):
                self.loadmesh(meshfname)
            else:
                print('Specified mesh file not found -- creating it')
                self.makemesh()
                self.savemesh(fromfile)

        self.p = None        # degree of finite elements used in mode calc
        self.a = None
        self.b = None
        self.V = None
        self.ngspmlset = None  # True if ngsolve pml set (then cant reuse mesh)
        self.X = None
        self.curvature = None

        self.setnondimmat(curvature=0)  # sets self.k and self.V
        L = self.fiber.rcore
        n0 = self.fiber.nclad
        super().__init__(self.mesh, L, n0)

    def __str__(self):

        s = '\nFiberMode Object: Nondimensional Computational Parameters:'
        s += '\n  Geometry consists of circular core (radius = 1), an annular'
        s += '\n  layer 1<r<R=%g, and another layer R<r<Rout=%g.'\
            % (self.R, self.Rout)
        s += '\n  Max mesh sizes: %g (core), %g (cladding), %g (pml)\n' \
            % (self.hcore, self.hclad, self.hpml)
        s += 'Physical Parameters:' + \
            '\n  Wavelength = %g meters' % (2*np.pi/self.fiber.ks)
        s += '\n  Refractive indices: %g (cladding), %g (core)' % \
            (self.fiber.nclad, self.fiber.ncore)
        if self.curvature is not None:
            s += '\n  Fiber bending curvature = %g' % self.curvature
        return s

    # FURTHER INITIALIZATIONS & SETTERS #####################################

    def makefibermode(self, fibername=None, R=None, Rout=None,
                      geom=None, h=4, hcore=None):

        self.fibername = fibername
        self.fiber = Fiber(fibername)

        if Rout is None:
            Rout = self.fiber.rclad / self.fiber.rcore
        if R is None:
            R = (Rout+1)/2
        if R < 1 or R > Rout:
            raise ValueError('Set R between 1 and Rout')
        self.R = R
        self.Rout = Rout

        if hcore is None:
            hcore = h/10
        self.hcore = hcore
        self.hclad = h
        self.hpml = h

    def makemesh(self, refine=0):
        self.setstepindexgeom()  # sets self.geo
        ngmesh = self.geo.GenerateMesh()
        for i in range(refine):
            ngmesh.Refine()
        mesh = ng.Mesh(ngmesh)
        mesh.Curve(3)
        ng.Draw(mesh)
        self.mesh = mesh

    def loadfibermode(self, fbmfilename):
        print('Loading FiberMode object from file ', fbmfilename)
        f = np.load(fbmfilename)
        self.fibername = str(f['fibername'])
        self.hcore = float(f['hcore'])
        self.hclad = float(f['hclad'])
        self.hpml = float(f['hpml'])
        self.R = float(f['R'])
        self.Rout = float(f['Rout'])

        self.fiber = Fiber(self.fibername)
        self.setstepindexgeom()  # sets self.geo

    def loadmesh(self, meshfname):
        print('Loading mesh from file ', meshfname)
        self.mesh = ng.Mesh(meshfname)
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh.Curve(3)
        ng.Draw(self.mesh)

    def setstepindexgeom(self):
        geo = SplineGeometry()
        geo.AddCircle((0, 0), r=self.Rout,
                      leftdomain=1, rightdomain=0, bc='OuterCircle')
        geo.AddCircle((0, 0), r=self.R,
                      leftdomain=2, rightdomain=1, bc='cladbdry')
        geo.AddCircle((0, 0), r=1,
                      leftdomain=3, rightdomain=2, bc='corebdry')
        geo.SetMaterial(1, 'Outer')
        geo.SetMaterial(2, 'clad')
        geo.SetMaterial(3, 'core')

        geo.SetDomainMaxH(1, self.hpml)
        geo.SetDomainMaxH(2, self.hclad)
        geo.SetDomainMaxH(3, self.hcore)

        self.geo = geo

    def setnondimmat(self, curvature=12, bendfactor=1.28):
        """
        When a fiber of refractive index n is bent to have the
        input "curvature" (curvature = reciprocal of bending radius,
        since we assume bending along a perfect circle), the changed
        refratcive index is modeled by the formula

            nbent = n * (1 + (x * curvature/bendfactor))

        with "bendfactor" as input. This dimensional formula is used
        non-dimensionally below to set the internal data member "V",
        the non-dimensional coefficient function for the eigenproblem.
        """

        self.curvature = curvature
        self.bendfactor = bendfactor
        fib = self.fiber
        self.k = fib.ks

        if curvature == 0:
            V = fib.fiberV()
            self.V = CoefficientFunction([0, 0, -V*V])
        else:
            n = CoefficientFunction([fib.nclad, fib.nclad, fib.ncore])
            a = fib.rcore
            ka2 = (fib.ks * a) ** 2
            kan2 = ka2 * (fib.nclad ** 2)

            nbent = n * (1 + (ng.x * a * curvature/bendfactor))

            m = kan2 - ka2 * nbent * nbent
            self.V = CoefficientFunction([0, m, m])

    # MODE CALCULATORS AND RELATED FUNCTIONALITIES  #########################

    def Z2toX2(self, Z2, v=None):
        """Convert non-dimensional Z² values to non-dimensional X² values
        through the relation X² - Z² = V². """

        V = self.fiber.fiberV() if v is None else v
        Zsqr = np.array(Z2)
        Vsqr = V**2
        return Zsqr + Vsqr

    def X2toBeta(self, X2, v=None):
        """Convert non-dimensional X² values to dimensional propagation
        constants beta through the relation (ncore*k)² - (X/a)² = beta². """

        V = self.fiber.fiberV() if v is None else v
        a = self.fiber.rcore
        ks = V / (self.fiber.numerical_aperture() * a)
        Xsqr = np.array(X2)

        return np.sqrt((ks*self.fiber.ncore)**2 - Xsqr/a**2)

    def Z2toBeta(self, Z2, v=None):
        """Convert nondimensional Z² (input as "Z2") in the complex plane to
        complex propagation constant Beta. """

        return self.X2toBeta(self.Z2toX2(Z2, v=v), v=v)

    def guidedmodes(self, interval=None, p=3, nquadpts=20, seed=1,
                    nspan=15, verbose=True, tone=False, **feastkwargs):
        """
        Search for guided modes in interval=(left, right). If interval is None,
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

        In case of multitone, data for each tone wavelength is stored as
        nested lists in the order specified in 'self.fibername'.
        As an example, betas[k][i] give the i-th real-valued
        propagation constant for k-th tone wavelength.
        """

        V = self.fiber.fiberV(tone=tone)
        if tone:
            k = [self.fiber.ks] + [self.fiber.ke]
        else:
            V = [V]
            k = [self.fiber.ks]
        betas = []
        Zsqrs = []
        Y = None
        fmind = [0]
        self.p = p

        for vnum, kk in zip(V, k):

            self.V = CoefficientFunction([0, 0, -vnum*vnum])
            self.k = kk

            if interval is None:
                # We choose the interval for the nondimensional Z² variable
                # recalling that  for guided modes,
                #         (L k₀ nclad)² < (β L)² < (L k₀ ncore)²,
                # where L is the scaling factor used to nondimensionalize.
                # It follows that Z² = (a α₀)² = (a k₀ nclad)² - (a β)²
                # satisfies
                #         0 > Z² > (a k₀ nclad)² - (a k₀ ncore)² = -V².
                interval = (-vnum*vnum, 0)

            betas_, Zsqrs_, Y_ =  \
                super().selfadjmodes(interval=interval, p=p, seed=seed,
                                     nspan=nspan, npts=nquadpts,
                                     verbose=verbose)

            betas = np.append(betas, betas_)
            Zsqrs = np.append(Zsqrs, Zsqrs_)
            if Y is None:
                Y = Y_
            else:
                for ind in range(len(betas_)):
                    Y._mv.Append(Y_._mv[ind])
                Y.m += len(betas_)
            fmind.append(fmind[-1] + len(betas_))
            fmind.append(len(betas))
            self.firstmodeindex = fmind
            self.X = Y.fes

        return betas, Zsqrs, Y

    def name2indices(self, betas, maxl=9, delta=None, tone=False):
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

        def construct_names(vnum, β):
            """
            constructs and saves LP names of propagation constants in β
            INPUTS:
                vnum: V-number in float
                β   : a numpy array containing propagation constants
            OUTPUTS:
                name2ind, exact: see self.name2indices docstring.
            """

            lft = self.Z2toBeta(0, v=vnum)  # βs must be in (lft, rgt)
            rgt = self.Z2toBeta(-vnum*vnum, v=vnum)
            # roughly identify simple and multiple ew approximants
            sm, ml = splitzoom.simple_multiple_zoom(lft, rgt, β,
                                                    delta=delta)

            name2ind = {}
            exact = -np.ones_like(β)

            # l=0 case should be simple eigenvalues:
            activesimple = np.arange(len(sm['index']))
            LP0 = self.fiber.XtoBeta(
                self.fiber.propagation_constants(0, v=vnum), v=vnum)
            b = β[sm['index']]
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
            for ll in range(1, maxl):
                LPl = self.fiber.XtoBeta(
                    self.fiber.propagation_constants(ll, v=vnum), v=vnum)
                for m in range(len(LPl)):
                    ind = np.argmin(abs(LPl[m]-ctrs[activemultiple]))
                    i2beta_a = ml['index'][activemultiple[ind]][0]
                    i2beta_b = ml['index'][activemultiple[ind]][1]
                    name2ind['LP' + str(ll) + str(m+1)+'_a'] = i2beta_a
                    name2ind['LP' + str(ll) + str(m+1)+'_b'] = i2beta_b
                    exact[i2beta_a] = LPl[m]
                    exact[i2beta_b] = LPl[m]
                    activemultiple = np.delete(activemultiple, ind)
                    if len(activemultiple) == 0:
                        return name2ind, exact
            return name2ind, exact

        V = self.fiber.fiberV(tone=tone)
        if tone:
            # in multitone, data will be stored in a list
            name2ind, exact = [], []
            for i, v in enumerate(V):
                betaslice = betas[self.firstmodeindex[i]:
                                  self.firstmodeindex[i+1]]
                n2i, ex = construct_names(v, betaslice)
                name2ind.append(n2i)
                exact.append(ex)
        else:
            name2ind, exact = construct_names(V, betas)
        return name2ind, exact

    # BENT MODES ############################################################

    def bentmode(self, curvature, radiusZ, centerZ, p,
                 bendfactor=1.28, **kwargs):

        self.setnondimmat(curvature=curvature, bendfactor=bendfactor)

        z, _, y, P, _, _ = self.leakymode(p, radiusZ, centerZ, **kwargs)

        print('Nonlinear eigenvalues in nondimensional Z-plane:\n', z)
        betas = self.ZtoBeta(z)
        print('Physical propagation constants:\n', betas)
        return betas, z, y, P

    # INTERPOLATED MODES ####################################################

    def interpmodes(self, p):
        """
        Return interpolated modes as an NGvecs object
        and propagation constants as a list for supported fibers.

        Nufern Yb-doped: 4 modes and betas
        Nufern Tm-doped: 2 modes and betas
        LLMA Yb-doped: 23 modes and betas
        """
        self.p = p
        self.X = H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)

        if self.fibername == 'LLMA_Yb':
            simple = list(range(4))
            multi = [list(range(1, 9)), list(range(1, 7)), list(range(1, 5)),
                     list(range(1, 2))]
        elif self.fibername == 'Nufern_Yb':
            simple = list(range(2))
            multi = [list(range(1, 3))]
        elif self.fibername == 'Nufern_Tm':
            simple = list(range(1))
            multi = [list(range(1, 2))]
        else:
            errmsg = 'Interp. modes not available for {}'.format(
                self.fibername)
            raise NotImplementedError(errmsg)

        phi, β, n2i = self.modepropn2i(simple, multi)

        gf = ng.GridFunction(self.X)
        n, m = len(gf.vec), len(phi)
        y = np.zeros((n, m), dtype=complex)
        for j, f in enumerate(phi):
            gf = ng.GridFunction(self.X)
            gf.Set(f)
            y[:, j] = gf.vec.FV().NumPy()[:]
        Y = NGvecs(self.X, m)
        Y.fromnumpy(y)
        return β, n2i, Y

    def modepropn2i(self, simple, multi):
        """
        INPUTS:
        simple: list of 'm' indices for simple modes ('l'=0)
        multi: nested list of 'l' indices for multiple modes,
               where 'm' is implied by the ordering of sublists.

        OUTPUTS:
        modes: CoefficientFunctions for the fiber modes
        betas: Propagation constants
        name2ind: A 'name to index' dict which places propagation
                  constants in descending order.
        """
        simple_pairs = [self.interpmodeLP(0, i) for i in simple]
        simple_names = ['LP0{}'.format(i+1) for i in simple]
        multi_pairs = [self.interpmodeLP(j, i) for i, lst in enumerate(multi)
                       for j in lst]
        multi_names = ['LP{}{}'.format(j, i+1) for i, lst in enumerate(multi)
                       for j in lst]
        betas, modes = zip(*(simple_pairs+multi_pairs))
        triples = sorted(list(zip(betas, modes, simple_names+multi_names)),
                         reverse=True)
        betas, modes, names = zip(*triples)  # lists ordered by betas
        name2ind = dict(zip(names, range(len(names))))
        return modes, betas, name2ind

    def interpmodeLP(self, ll, m):
        """
        Return un-normalized LP(ll,m) "mode" of the fiber as an NGSolve
        CoefficientFunction and its corresponding propagation
        constant "beta" when calling:

           beta, mode = fbm.interpmodeLP(ll, m)

        Note that l and m are both indexed to start from 0, so for
        example, the traditional LP_01 and LP_11 modes are obtained by
        calling LP(0, 0) and LP(1, 0), respectively.

        See also Fiber.visualize_mode(l, m).
        """

        X = self.fiber.propagation_constants(ll)

        if len(X) <= m:
            raise ValueError('For ll=%d, only %d fiber modes computed'
                             % (ll, len(X)))

        kappa = X[m] / self.fiber.rcore
        ncore, nclad = self.fiber.ncore, self.fiber.nclad
        k0 = self.fiber.ks
        beta = ng.sqrt(ncore*ncore*k0*k0 - kappa*kappa)
        gamma = ng.sqrt(beta*beta - nclad*nclad*k0*k0)

        r = ng.sqrt(ng.x*ng.x + ng.y*ng.y)
        theta = ng.atan2(ng.y, ng.x)

        print('\nCOMPUTED LP(%1d,%d) MODE: ' % (ll, m) + '-'*49)
        print('  beta:      %20g' % (beta) +
              '{:>39}'.format('exact propagation constant'))

        # If NA=0, then return the Bessel mode of an empty waveguide:
        if abs(self.fiber.numerical_aperture()) < 1.e-15:
            print('  NA = 0, so further parameters are meaningless.\n')
            a0cf = jv(kappa*r*self.R, ll) * ng.cos(ll*theta)

            return beta, a0cf

        # For NA>0, define the guided mode piecewise:
        print('  variation: %20g' % (k0*abs(nclad-ncore)) +
              '{:>39}'.format('interval length of propagation consts'))
        Jkrcr = scf.jv(ll, kappa*self.fiber.rcore)
        Kgrcr = scf.kv(ll, gamma*self.fiber.rcore)
        print('  edge value:%20g'
              % (Jkrcr*scf.kv(ll, gamma*self.fiber.rclad)) +
              '{:>39}'.format('mode size at outer cladding edge'))
        print('  kappa:     %20g' % (kappa) +
              '{:>39}'.format('coefficient in BesselJ core mode'))
        print('  gamma:     %20g' % (gamma) +
              '{:>39}'.format('coefficient in BesselK cladding mode'))

        Jkr = jv(kappa*r*self.fiber.rcore, ll)
        Kgr = kv(gamma*r*self.fiber.rcore, ll)

        a0cf = IfPos(1 - r, Kgrcr*Jkr, Jkrcr*Kgr) * ng.cos(ll*theta)
        return beta, a0cf

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
    #  * Modes saved in file(s):         <prefix>_mde.npz for Feast modes
    #                       or           <prefix>_imde.npz for interp modes
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
                 R=self.R, Rout=self.Rout)

    def savemesh(self, fileprefix):

        meshfname = self.outfolder+'/'+fileprefix+'_msh.vol.gz'
        print('Writing mesh into:\n', meshfname)
        self.mesh.ngmesh.Save(meshfname)

    def savemodes(self, fileprefix, betas, Y,
                  saveallagain=True, name2ind=None, exact=None,
                  interp=False, tone=False):
        """ Convert Y to numpy and save in npz format. """

        if saveallagain:
            self.savefbm(fileprefix)
            self.savemesh(fileprefix)

        y = Y.tonumpy()

        if os.path.isdir(self.outfolder) is not True:
            os.mkdir(self.outfolder)
        suffix = '_imde.npz' if interp else '_mde.npz'
        fullname = self.outfolder+'/'+fileprefix+suffix
        print('Writing modes into:\n', fullname)
        if tone:
            np.savez(fullname, fibername=self.fibername,
                     hcore=self.hcore, hclad=self.hclad, hpml=self.hpml,
                     p=self.p, R=self.R, Rout=self.Rout,
                     betas=betas, y=y,
                     exactbetas=exact, name2ind=name2ind,
                     firstmodeindex=self.firstmodeindex)
        else:
            np.savez(fullname, fibername=self.fibername,
                     hcore=self.hcore, hclad=self.hclad, hpml=self.hpml,
                     p=self.p, R=self.R, Rout=self.Rout,
                     betas=betas, y=y,
                     exactbetas=exact, name2ind=name2ind)

    def checkload(self, f):
        """Check if the loaded file has expected values of certain data"""

        for member in {'fibername', 'hcore', 'hclad', 'hpml',
                       'R', 'Rout'}:
            print('  From file:', member, '=', f[member])
            assert self.__dict__[member] == f[member], \
                'Load error! Data member %s does not match!' % member

    def loadmodes(self, modefile, tone=False):
        """Load modes from "outputs/modefile" (filename with extension)"""

        fname = self.outfolder+'/'+modefile
        if os.path.isfile(fname):
            print('Loading modes from:\n ', fname)
            f = np.load(fname, allow_pickle=True)
            self.checkload(f)
            self.p = int(f['p'])
            print('  Degree %d modes found in file' % self.p)
            self.X = H1(self.mesh, order=self.p, dirichlet='OuterCircle',
                        complex=True)
            y = f['y']
            betas = f['betas']
            if tone:
                n2i = f['name2ind'].tolist()
                self.firstmodeindex = f['firstmodeindex'].tolist()
            else:
                n2i = f['name2ind'].item()
            m = y.shape[0]
            Y = NGvecs(self.X, m)
            Y.fromnumpy(y)
        else:
            print('Specified modes file not found -- creating it')
            fibername, p, interp = _extract_fbname_and_p(modefile)
            if interp:
                betas, n2i, Y = self.interpmodes(p=p)
                self.savemodes(fibername+'_p' + str(p), betas, Y,
                               saveallagain=False, name2ind=n2i,
                               exact=betas, interp=True)
            else:
                betas, zsqrs, Y = self.guidedmodes(p=p, nspan=50, tone=tone)
                n2i, exbeta = self.name2indices(betas, maxl=9, tone=tone)
                self.savemodes(fibername+'_p' + str(p), betas, Y,
                               saveallagain=False, name2ind=n2i,
                               exact=exbeta, tone=tone)
        return betas, Y, n2i

    def makeguidedmodelibrary(self, maxp=5, maxl=9, delta=None,
                              nspan=15, interp=False, tone=False):
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
            if interp:
                betas, n2i, Y = self.interpmodes(p=p)
                print('Physical propagation constants:\n', betas)
                self.savemodes(fprefix+'_p' + str(p), betas, Y,
                               saveallagain=False, name2ind=n2i,
                               exact=betas, interp=True)
            else:
                betas, zsqrs, Y = self.guidedmodes(p=p, nspan=nspan, tone=tone)
                print('Physical propagation constants:\n', betas)
                print('Computed non-dimensional Z-squared values:\n', zsqrs)
                n2i, exbeta = self.name2indices(betas, maxl=maxl, delta=delta,
                                                tone=tone)
                self.savemodes(fprefix+'_p' + str(p), betas, Y,
                               saveallagain=False, name2ind=n2i, exact=exbeta,
                               tone=tone)


# END OF CLASS DEFINITION ###################################################

# Helper methods


def _extract_fbname_and_p(fn):
    """
    Extract the fibername, polynomial order and mode type
    from a mode filename
    """
    pfx = None
    sfxs = ['_mde.npz', '_imde.npz']
    for sfx in sfxs:
        if sfx in fn:
            pfx = fn[:fn.find(sfx)]
            break
    if pfx is None:
        return None
    parts = pfx.split('_')
    fibername = '_'.join(parts[:-1])
    p = int(parts[-1][1:])
    interp = (sfx == sfxs[1])
    return fibername, p, interp

# MODULE END #############################################################
