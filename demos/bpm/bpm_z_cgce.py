"""
CONVERGENCE OF z-DISCRETIZATION IN BPM


1. BPM approximates the function u solving

(a)   2i kt L² du/dz =  -Δu + Vu + L²(kt² - k²n₀²)u

by the function uₕ solving the finite element formulation

(b)   2i kt L² (uₕ', vₕ) =  (∇uₕ, ∇vₕ) + (V uₕ, vₕ) + (kt² - k²n₀²) L² (uₕ, vₕ)

for all vₕ in the finite element space.

2. It can be easily verified that an exact solution for the finite element
system (b) is

(c)   uₕ(x, y, z) = φₕ(x, y) exp( ½ i (β² - kt²) z / kt)

where φₕ(x, y) is the discrete eigenmode satisfying

(d)   (∇φₕ, ∇vₕ) + (V φₕ, vₕ) = β² (φₕ, vₕ).

3. After discretizing uₕ', the semidiscrete solution in (c) can be compared
to the fully discrete computed solution to check convergence of the
z-discretization. (This does not take into account the error in the
h-discretization of φₕ, but that is not the focus here.)

"""

from fibermode import StepIndex, BPM
import ngsolve as ng
import numpy as np
from prettytable import PrettyTable

# Set up eigenmode and BPM

p = 2
fb = StepIndex(fibername="Nufern_Yb", curveorder=p)
betas, zsqrs, Y = fb.guidedmodes(p=p, niterations=200, verbose=False)
m = 0  # choose m-th mode as φₕ

bpm = BPM(fb)
kt = bpm.kt_default()
Z = 0.05  # total fiber length (meters)

N = 50  # number of z-points per esrtimated beat length
dz = abs(2 * ng.pi / (betas[m] - kt)) / N
bpm.setupCrankNicolson(dz, p, kt=kt)
nsteps = int(Z / dz)


def exactu(z):  # Implement the above formula (c)
    return u_initial * ng.exp(1j * 0.5 * (betas[m]**2 - kt**2)*z/kt)


# Propagate using Crank-Nicolson with various dz values

nrefine_z = 5  # number of experiments, each time halving dz
u_initial = Y[m]
errors = []

for i in range(nrefine_z):

    u = bpm.propagateCrankNicolson(u_initial, nsteps)

    z = bpm.dz * nsteps
    uex = exactu(z)
    errL2 = ng.sqrt(ng.Integrate((uex - u)*ng.Conj(uex - u), fb.mesh).real)
    print('After %6d steps of BPM(dz=%4e), L² error = %.6f' %
          (nsteps, bpm.dz, errL2))
    errors.append(errL2.real)

    bpm.dz /= 2
    nsteps *= 2

errors = np.array(errors)

# Report convergence rates in a table

t = PrettyTable()
col = ['dz', 'L² error', 'rate']
h0col = ['%g' % dz]
t.add_column(col[0], h0col + [h0col[0] + '/' + str(2**i)
                              for i in range(1, len(errors))])
t.add_column(col[1], ['%.12f' % e for e in errors])
t.add_column(col[2], ['*'] +
             ['%1.2f' % r
              for r in np.log(errors[:-1]/errors[1:])/np.log(2)])
print(t)
