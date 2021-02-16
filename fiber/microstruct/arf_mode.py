"""
Computing some modes of ARF
"""

from arf import ARF

# Compute modes using degree p finite elements
a = ARF()
p = 2
Zs, Ys, betas, P = a.polyeig(p=p, ctrs=(2.24,), radi=(0.05,))
print('Zs =', Zs)
print('betas =', betas)
Ys[0].draw()

# Save modes into file (together with mesh and ARF object)
a.savemodes('arfLP01_p%d' % p, Ys[0], p, betas,
            {'method': 'polyeig',
             'Zs': Zs,
             'name': 'LP01'})

# Refine mesh once, re-compute, and save
refine = 1
a.refine()
Zs, Ys, betas, P = a.polyeig(p=p, ctrs=(2.24,), radi=(0.05,))
a.savemodes('arfLP01_p%d_r%d' % (p, refine), Ys[0], p, betas,
            {'method': 'polyeig',
             'Zs': Zs,
             'name': 'LP01',
             'refine': 1})

# To load any of these solution files, follow arf_load.py
