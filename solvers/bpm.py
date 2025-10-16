import ngsolve as ng
from ngsolve import dx, grad


class BPM():

    def __init__(self, fiber, dz, kt=None, p=None):
        """
        Parameters:
         fiber : any derived class of ModeSolver representing a fiber
            dz : propagation step size in the longitudinal direction
            kt : estimated effective wavenumber of propagation
             p : polynomial degree of the transverse FE discretization
        """
        self._dz = dz
        self._fiber = fiber
        if kt is None:
            self._kt = self.fiber.fiber.ks * self.fiber.n0
        else:
            self._kt = kt
        if p is None:
            self._p = self.fiber.curveorder
        else:
            self._p = p
        self.setCrankNicolsonPropagator()

    def setCrankNicolsonPropagator(self):

        self.X = ng.H1(self.fiber.mesh, order=self.p,
                       dirichlet='OuterCircle', complex=True)
        u, v = self.X.TnT()
        A = ng.BilinearForm(self.X)
        A += grad(u) * grad(v) * dx + self.fiber.V * u * v * dx
        A += self.fiber.L**2 * (self.kt**2 - (self.fiber.k*self.fiber.n0)**2) \
            * u * v * dx
        B = ng.BilinearForm(self.X)
        B += u * v * dx

        with ng.TaskManager():
            A.Assemble()
            B.Assemble()
        self.A = A
        self.B = B

        R = B.mat.CreateMatrix()
        L = B.mat.CreateMatrix()
        c = self.dz / (4j * self.kt * self.fiber.L**2)

        R.AsVector().data = B.mat.AsVector() + c * A.mat.AsVector()
        L.AsVector().data = B.mat.AsVector() - c * A.mat.AsVector()
        Linv = R.Inverse(self.X.FreeDofs())
        self.P = Linv @ R

    @property
    def fiber(self):
        return self._fiber

    @fiber.setter
    def fiber(self, fiber):
        raise AttributeError("Remake BPM object if you must change fiber")

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, dz):
        self._dz = dz
        # update all data structures that depend on dz - TO DO

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        # update all data structures that depend on p - TO DO

    @property
    def kt(self):
        return self._kt

    @kt.setter
    def kt(self, kt):
        self._kt = kt
        # update all data structures that depend on kt - TO DO
