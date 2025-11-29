import ngsolve as ng
from ngsolve import dx
from .modesolver import ModeSolver
import logging


class BPM():

    def __init__(self, fiber):
        """
        Beam Propagation Method (BPM) for approximately propagating fields
        in a given fiber. The constructor input "fiber" must be an object
        of a derived class of class ModeSolver.

        This class provides facilities for solving for the envelop
        field u(x, y, z) such that

              U(x, y, z) = u(x, y, z) exp(i kt z)

        approximately solves the Helmholtz equation

              ΔU + k² n² U = 0

        in the fiber, for some effective longitudinal propagation constant kt.
        Assuming that L² d²u/dz² can be neglected, we obtain the BPM equation

              2i kt L² du/dz =  -Δu + Vu + L²(kt² - k²n₀²)u

        where V, L, k, and n₀ are properties of the fiber (as described
        in ModeSolver class). Facilities are provided for transverse
        discretization of the right hand side above and to step in z.
        """

        if not issubclass(type(fiber), ModeSolver):
            raise ValueError('Input fiber must have attributes of ModeSolver')
        self._fiber = fiber
        self.propagator = None

    def resetPropagator(self, dz, p, kt):

        if self.propagator == 'Crank-Nicolson':
            self.setupCrankNicolson(dz, p, kt)
        elif self.propagator is None:
            raise AttributeError('Set up a field propagator first')
        else:
            raise AttributeError('What propagator?')

    def kt_default(self):
        """Return default value of longitudinal propagation constant,
        kt = k * n₀, using the fiber properties."""

        return self.fiber.fiber.ks * self.fiber.n0

    def setupCrankNicolson(self, dz, p, kt=None, pml={}):
        """
        Sets up data structures for solving the Crank-Nicolson scheme

           2i * kt * L² * B (u[n+1] - u[n])/dz = A (u[n+1] + u[n])/2

        where B is the mass matrix and A is the stiffness matrix of the
        transverse operator A = -∇² + V + L² (kt² - (k n₀)²)
        where V, L, k, n₀ are properties of the fiber, with or
        without PML. Internal attributes are set as follows:
        _X: Lagrange FE space,
        _Linv: sparse inverse of B - dz/(4i kt L²) A,
        _R: matrix of B + dz/(4i kt L²) A
            (so propagation step is u[n+1] = Linv * R * u[n]),
        _M: mass matrix (without PML wvene when PML is used).

        PARAMETERS:
        dz:  The propagation step size in the z direction
        p:   Degree of finite elements for the spatial discretization
             on the mesh from the fiber.
        kt:  The effective longitudinal propagation constant. If None,
             the value kt = k * n₀ from the fiber is used.
        pml: If pml is an empty dict, use no PML.
             If pml['type']=='auto', use ngsolve PML on cross section
                of strength pml['alpha'].
        """
        self.propagator = 'Crank-Nicolson'
        self._dz = dz
        if kt is None:
            self._kt = self.kt_default()
        else:
            self._kt = kt
        self._p = p
        self._pml = pml
        self._M = None  # mass matrix

        # Get matrices H of (grad u, grad v) + (V u, v) and B of (u, v):

        if len(pml) == 0:
            H, B, X = self._fiber.selfadjsystem(self._p)
            self._M = B.mat
        elif pml['type'] == 'auto':
            H, B, X = self._fiber.autopmlsystem(self._p, alpha=pml['alpha'])
            # B is not mass matrix due to PML, so make mass matrix separately,
            # assuming that PML is unset
            assert not self._fiber.ngspmlset, \
                'Computing mass matrix with PML set will give wrong results'
            u, v = X.TnT()
            with ng.TaskManager():
                m = ng.BilinearForm(u*v*dx)
                m.Assemble()
                self._M = m.mat
        else:
            NotImplementedError('BPM asks for pml type not yet implemented')

        with ng.TaskManager():
            # Make cA = c * A, where A = H + a * B, with numbers c, a below:
            cA = H.mat.CreateMatrix()
            a = self.fiber.L**2*(self.kt**2 - (self.fiber.k*self.fiber.n0)**2)
            c = self.dz / (4j * self.kt * self.fiber.L**2)
            cA.AsVector().data = \
                c * (H.mat.AsVector() + a * B.mat.AsVector())
            R = cA.CreateMatrix()  # R = B + dz/(4i kt L²) A
            L = cA.CreateMatrix()  # L = B - dz/(4i kt L²) A.
            R.AsVector().data = B.mat.AsVector() + cA.AsVector()
            L.AsVector().data = B.mat.AsVector() - cA.AsVector()
        Linv = L.Inverse(X.FreeDofs())

        self._X = X
        self._Linv = Linv  # store inverse of B - dz/(4i kt L²) A.
        self._R = R        # store sparse mat B + dz/(4i kt L²) A.

    def propagateCrankNicolson(self, u0, nsteps,
                               save_every=None, zero_stop=1e-16):
        """Use

          u = bpm.propagateCrankNicolson(u0, nsteps),   or

          u, u_samples, z_samples, p_samples = \
              bpm.propagateCrankNicolson(u0, nsteps, save_every=10)

        to propagate input field given in GridFunction "u0" for "nsteps"
        of size "dz" and return the output field as another GridFunction "u".

        If "save_every"=None (the default) and only the last step is output.

        If "save_every"=n, then at every n-th z-step the solution is saved
        and combined into a multi-dimensional output GridFunction "u_samples",
        along with lists "z_samples" of corresponding z-values and
        "p_samples" of power(z) = ∫ |u(z)|² dx dy at each z-value.
        Additionally, when power(z) < zero_stop, the iterations are stopped.
        """

        if self.propagator != 'Crank-Nicolson':
            raise ValueError('Crank-Nicolson propagator not set!')
        if self._X.ndof != len(u0.vec):
            raise ValueError('Input does not have the right size!')

        u = ng.GridFunction(self._X)
        work = u.vec.CreateVector()
        u.vec.data = u0.vec
        if save_every is not None:
            logging.basicConfig(format='%(message)s', level=logging.WARNING)
            u_samples = ng.GridFunction(self._X, multidim=0)
            u_samples.AddMultiDimComponent(u0.vec)
            z_samples = [0.0]
            Mu = u.vec.CreateVector()
            Mu.data = self._M * u.vec
            p_samples = [ng.InnerProduct(Mu, u.vec).real]

        with ng.TaskManager():
            for step in range(nsteps):
                work.data = self._R * u.vec
                u.vec.data = self._Linv * work

                if save_every is not None:
                    if (step+1) % save_every == 0:
                        u_samples.AddMultiDimComponent(u.vec)
                        z_samples.append(step*self._dz)
                        Mu.data = self._M * u.vec
                        power = ng.InnerProduct(Mu, u.vec).real
                        p_samples.append(power)
                        if power < zero_stop:
                            logging.warning(
                                f'Stopping at step {step+1}, '
                                f'z={step*self._dz:.3f}, '
                                f'power={power:.3e} < {zero_stop:.3e}')
                            break

        if save_every is not None:
            return u, u_samples, z_samples, p_samples
        else:
            return u

    @property
    def fiber(self):
        return self._fiber

    @fiber.setter
    def fiber(self, fiber):
        raise AttributeError('Remake BPM object if you must change fiber')

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, dz):
        self.resetPropagator(dz, self.p, self.kt)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self.resetPropagator(self.dz, p, self.kt)

    @property
    def kt(self):
        return self._kt

    @kt.setter
    def kt(self, kt):
        self.resetPropagator(self.dz, self.p, kt)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        raise AttributeError('Space can only be set by propagator')
