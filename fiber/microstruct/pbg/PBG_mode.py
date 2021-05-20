from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

if __name__ == '__main__':

    A = PBG(params)

    center = 1.242933-2.471929e-09j
    radius = .01
    p = 2

    z, y, yl, beta, P, _ = A.leakymode(p, rad=radius, ctr=center,
                                       alpha=A.alpha, niterations=50, npts=4,
                                       nspan=2)

    y.draw()
