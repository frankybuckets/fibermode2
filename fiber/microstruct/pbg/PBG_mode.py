
from fiberamp.fiber.pbg import PBG
from fiberamp.fiber.pbg.fiber_param_dict import fiber_param_dict

A = PBG(fiber_param_dict)

z2, y2, yl2, beta2, P2, _ = A.leakymode(2, ctr=3.56545193-0.00913029j,
                                        rad=.01)
