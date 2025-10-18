from fibermode import StepIndex, BPM
import ngsolve as ng


def test_bpm_propagation():
    """
    Does BPM propagation preserve guided modes?
    """

    p = 2
    fb = StepIndex(fibername="Nufern_Yb", curveorder=p)
    betas, zsqrs, Y = fb.guidedmodes(p=p, stop_tol=1e-14, niterations=200)
    diff = []

    bpm = BPM(fb)

    for i in range(len(betas)):
        bpm.setupCrankNicolson(0.1, p, kt=betas[i])
        u_initial = Y[i]
        u = bpm.propagateCrankNicolson(u_initial, 10)
        diff.append(ng.Norm(u.vec - u_initial.vec))
        print("Case", i, " error", diff[-1])

    print("Max difference after propagation:", max(diff))
    assert max(
        diff) < 1e-12, "BPM propagation deviates too much from initial mode."


test_bpm_propagation()
