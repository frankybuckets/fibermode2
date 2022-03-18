from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params

if __name__ == '__main__':

    A = PBG(params)

    zcenter = 2
    radius = 1
    p = 2

    betas, zsqrs, E, phi, R = \
        A.leakyvecmodes(rad=radius, ctr=zcenter**2,
                        alpha=A.alpha, p=p,
                        quadrule='ellipse_trapez_shift', rhoinv=.8,
                        niterations=12, npts=6, stop_tol=1e-8,
                        nspan=5, nrestarts=0)

    E.draw(name='E')
    phi.draw(name='phi')
