"""
Computing some modes of ARF 
"""

from arf import ARF

a = ARF()

_, Ys, betas = a.polyeig(p=3)

print('betas =', betas)

Ys[0].draw()
