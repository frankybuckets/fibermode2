"""
Computing some modes of a 6-tube AFR structure in Poletti's paper
"""

from arf import ARF

a = ARF(name='poletti', freecapil=False)
p = 3         # finite element degree
a.refine()

# compute:
Zs, Ys, Yls, betas, P, _, _, _ =  \
    a.polyeig(p=p,
              #    LP01,   LP11,  LP21   LP02
              ctrs=(2.24,  3.57,  4.75,  5.09),
              radi=(0.02,  0.02,  0.02,  0.02),
              npts=4, alpha=5,
              eta_tol=1.e-12, stop_tol=1e-12)
print('All computes eigenvalues:\n')
print(' Zs =', Zs)
print(' betas =', betas)

# visualize and / or save into file:

Ys[-1].draw()  # visualize in netgen window

# save the modes to file
modenames = ['LP01', 'LP11', 'LP21', 'LP02']
fileprefixes = ['tmp_arf{0}_p{1}'.format(name, p) for name in modenames]

for k in range(len(fileprefixes)):
    solverparams = {'method': 'polyeig', 'Zs': (Zs[k],),
                    'name': modenames[k]}
    a.savemodes(fileprefixes[k], Ys[k], p, (betas[k],), (Zs[k],),
                solverparams, arfpickle=True)
