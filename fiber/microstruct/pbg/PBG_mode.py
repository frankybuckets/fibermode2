
from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr7cr1 import params

A = PBG(params)

z2, y2, yl2, beta2, P2, _ = A.leakymode(2, ctr=3.56545193-0.00913029j,
                                        rad=.01)
