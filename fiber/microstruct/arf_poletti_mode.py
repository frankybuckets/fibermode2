"""
Computing some modes of a 6-tube ARF structure in Poletti's paper
"""

from arf import ARF

a = ARF(name='poletti', freecapil=False)
p = 3         # finite element degree
# a.refine()

z, y, yl, beta, P, _ = a.leakymode(p, ctr=2.24, rad=0.01, alpha=5)

y.draw()

# save results and a dict of solver paramaters
solverparams = dict(p=p, ctr=2.24, rad=0.01, alpha=5)
a.savemodes('tmp_arf_LP01', y, p, beta, z, solverparams, arfpickle=True)


#
# NOTE ------------------------------------------------
#
# For wavelength = 1800nm, we found these work:
#               #    LP01,   LP11,  LP21   LP02
#               ctrs=(2.24,  3.57,  4.75,  5.09),
#               radi=(0.02,  0.02,  0.02,  0.02),
