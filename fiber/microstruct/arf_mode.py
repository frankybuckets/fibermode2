"""
Computing some modes of ARF
"""

from arf import ARF


a = ARF()

p = 2         # finite element degree
a.refine()    # refine the default mesh once (comment out if short of memory)

Zs, Ys, Yls, betas, P, _, _ =  \
    a.polyeig(p=p, ctrs=(5.1,), radi=(0.05,), npts=6)

print('Zs =', Zs)
print('betas =', betas)
Ys[0].draw()  # visualize in netgen window

a.savemodes('arfLP02_p%d' % p, Ys[0], p, betas, Zs,
            {'method': 'polyeig', 'Zs': Zs,
             'name': 'LP02'}, arfpickle=True)
