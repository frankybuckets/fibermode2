"""
Computing some modes of ARF
"""

from arf import ARF

name = 'kolyadin'
freecapil = False
a = ARF(name=name, freecapil=freecapil)

p = 2         # finite element degree
a.refine()    # refine the default mesh once (comment out if short of memory)

Zs, Ys, Yls, betas, P, _, _ =  \
    a.polyeig(p=p, ctrs=(2.35,), radi=(0.05,), npts=6)

print('Zs =', Zs)
print('betas =', betas)
Ys[0].draw()  # visualize in netgen window

a.savemodes('arfLP01_p%d' % p, Ys[0], p, betas, Zs,
            {'method': 'polyeig', 'Zs': Zs,
             'name': 'LP02'}, arfpickle=True)
