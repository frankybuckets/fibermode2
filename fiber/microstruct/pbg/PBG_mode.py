import netgen.gui
from ngsolve import Draw, GridFunction
from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

if __name__ == '__main__':

    folder = 'your/output/folder'

    A = PBG(params)

    center = 1.5
    radius = 1
    p = 2

    z, y, yl, beta, P, _ = A.leakymode(p, rad=radius, ctr=center,
                                       alpha=A.alpha,
                                       quadrule='ellipse_trapez_shift',
                                       rhoinv=.9, niterations=50, npts=8,
                                       nspan=6, nrestarts=0)
    for i in range(len(y)):
        a = GridFunction(y.fes, name='sol_' + str(i))
        a.vec.data = y._mv[i]
        Draw(a)

    A.savemodes('pbg', folder, y, p, beta, z)
