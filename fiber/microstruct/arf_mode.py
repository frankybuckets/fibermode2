"""
Computing some modes of ARF
"""

from arf import ARF

name = 'kolyadin'
freecapil = False
a = ARF(name=name, freecapil=freecapil)

p = 3         # finite element degree
# a.refine()    # refine the default mesh once (comment out if short of memory)

Zs, Ys, Yls, betas, P, _, _ =  \
    a.polyeig(p=p, ctrs=(2.34,), radi=(0.01,), npts=4, alpha=5,
              eta_tol=1.e-12, stop_tol=1e-12)

print('Zs =', Zs)
print('betas =', betas)
Ys[0].draw()  # visualize in netgen window

a.savemodes('tmp_arfLP01_p%d' % p, Ys[0], p, betas, Zs,
            {'method': 'polyeig', 'Zs': Zs,
             'name': 'LP02'}, arfpickle=True)
