from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

A = PBG(params)

center = 5.49955777-0.06779686j
radius = .01
p = 7

z, y, yl, beta, P, _ = A.leakymode(p, rad=radius, ctr=center, alpha=A.alpha,
                                   niterations=50, npts=4, nspan=2)
