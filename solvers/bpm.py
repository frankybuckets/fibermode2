import ngsolve as ng
from ngsolve import dx, grad


class BPM():

    def __init__(self, fiber):
        """
        Beam Propagation Method (BPM) solver for propagation in the input
        "fiber", which is any object of a derived class of ModeSolver.
        """
        self._fiber = fiber
        fiber.needs()
        self.propagator = None

    def resetPropagator(self, dz, p, kt):
        if self.propagator == 'Crank-Nicolson':
            self.setupCrankNicolson(dz, p, kt)
        elif self.propagator is None:
            return
        else:
            raise AttributeError('Set up a propagator first')

    def setupCrankNicolson(self, dz, p, kt=None):
        """
        Set up data structures for solving the Crank-Nicolson scheme
           2i * kt * L² * B (u[n+1] - u[n])/dz = A (u[n+1] + u[n])/2
        where B is the mass matrix and A the stiffness matrix of the
        transverse operator A = -∇² + V + L² (kt² - (k n₀)²)
        where V, L, k, n₀ are properties of the fiber. The spatial
        discretization uses finite elements of order "p" on the mesh
        from the fiber. The propagation step size is "dz".
        """
        self.propagator = 'Crank-Nicolson'
        self._dz = dz
        if kt is None:
            self._kt = self.fiber.fiber.ks * self.fiber.n0
        else:
            self._kt = kt
        self._p = p
        self._X = ng.H1(self.fiber.mesh, order=self.p,
                        dirichlet='OuterCircle', complex=True)
        u, v = self._X.TnT()
        A = ng.BilinearForm(self._X)
        A += grad(u) * grad(v) * dx + self.fiber.V * u * v * dx
        A += self.fiber.L**2 * (self.kt**2 - (self.fiber.k*self.fiber.n0)**2) \
            * u * v * dx
        B = ng.BilinearForm(self._X)
        B += u * v * dx
        with ng.TaskManager():
            A.Assemble()
            B.Assemble()
            # Make R = B + dz/(4i kt L²) A and
            #      L = B - dz/(4i kt L²) A.
            R = B.mat.CreateMatrix()
            L = B.mat.CreateMatrix()
            c = self.dz / (4j * self.kt * self.fiber.L**2)
            R.AsVector().data = B.mat.AsVector() + c * A.mat.AsVector()
            L.AsVector().data = B.mat.AsVector() - c * A.mat.AsVector()
        Linv = R.Inverse(self._X.FreeDofs())

        self._Linv = Linv  # store inverse of B - dz/(4i kt L²) A.
        self._R = R        # store sparse matrix B + dz/(4i kt L²) A.

    def propagateCrankNicolson(self, u0, nsteps):
        """
        Propagate input field given in GridFunction "u0" for "nsteps"
        of size "dz" and return the output as another GridFunction.
        """

        if self.propagator != 'Crank-Nicolson':
            raise ValueError('Crank-Nicolson propagator not set!')
        u = ng.GridFunction(self._X)
        if self._X.ndof != len(u0.vec):
            raise ValueError('Input does not have the right size!')

        work = u.vec.CreateVector()
        u.vec.data = u0.vec
        for step in range(nsteps):
            work.data = self._R * u.vec
            u.vec.data = self._Linv * work
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
