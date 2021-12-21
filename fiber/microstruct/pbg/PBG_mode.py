from ngsolve import GridFunction
from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

if __name__ == '__main__':

    # folder = 'your/output/folder'

    A = PBG(params)

    center = 1.5
    radius = 1
    p = 3

    z, y, yl, beta, P, _ = A.leakymode(p, rad=radius, ctr=center,
                                       alpha=A.alpha, stop_tol=1e-11,
                                       quadrule='ellipse_trapez_shift',
                                       rhoinv=.9, niterations=50, npts=8,
                                       nspan=6, nrestarts=0)
    for i in range(len(y)):
        a = GridFunction(y.fes, name='sol_' + str(i))
        a.vec.data = 1.5e-6 * y._mv[i]

    # A.savemodes('pbg', folder, y, p, beta, z)
