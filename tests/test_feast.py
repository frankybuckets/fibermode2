from fibermode import StepIndex
import ngsolve as ng
from ngsolve import dx, grad
import numpy as np


def test_guided_residual():
    """
    Does StepIndex.guidedmode give modes with small residuals?
    """

    p = 2
    fb = StepIndex(fibername='Nufern_Yb', curveorder=p)
    betas, zsqrs, Y = fb.guidedmodes(p=p, stop_tol=1e-14, niterations=200)
    Z2 = ng.Vector(zsqrs)

    X = ng.H1(fb.mesh, order=p, dirichlet='OuterCircle', complex=True)
    u, v = X.TnT()
    A = ng.BilinearForm(X)
    A += grad(u) * grad(v) * dx + fb.V * u * v * dx
    B = ng.BilinearForm(X)
    B += u * v * dx
    with ng.TaskManager():
        A.Assemble()
        B.Assemble()

    t = ng.MultiVector(Y._mv[0], len(Y._mv))
    t[:] = A.mat * Y._mv - (B.mat * Y._mv).Scale(Z2)
    residuals = np.diag(abs(ng.InnerProduct(t, t).NumPy()))

    assert max(residuals) < 1e-11, \
        "Step-index guided modes are not accurate."
    print("Test passed: Guided modes have small residuals:\n", residuals)


test_guided_residual()
