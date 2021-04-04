"""
Computing some modes of a 6-tube ARF structure in Poletti's paper
"""

from arf import ARF

a = ARF(name='poletti', freecapil=False)
p = 2         # finite element degree

# Refine mesh if needed:
# a.refine()

# Solve for leaky mode using freq-dependent PML & polynomial eig
print('POLY PML' + '-'*60)
solverparams = dict(p=p, ctr=2.247, rad=0.001, alpha=5)
# p, ctr=2.247, rad=0.001, alpha=5)
z, y, yl, beta, P, _ = a.leakymode(**solverparams)
y.draw()

# Save results and a dict of solver paramaters
a.savemodes('tmp_arf_LP01', y, p, beta, z, solverparams, arfpickle=True)

# Try other solvers, like the standard PML & linear eig:
print('SMOOTH PML' + '-'*60)
z2, yy, yyl, beta2, P2 = \
    a.leakymode_smooth(p, centerZ2=5, radiusZ2=0.1, alpha=5)
print('AUTO PML' + '-'*60)
z3, y3, yl3, beta3, P3 = \
    a.leakymode_auto(p, centerZ2=5, radiusZ2=0.1, alpha=5)


#
# NOTE ------------------------------------------------
#
# For wavelength = 1800nm, we found these work:
#               #    LP01,   LP11,  LP21   LP02
#               ctrs=(2.24,  3.57,  4.75,  5.09),
#               radi=(0.02,  0.02,  0.02,  0.02),
