"""
Some mode computations on ARF model using a selfadjoint eigenproblem,
yet to be modified for proper resonance computation.
"""

from pyeigfeast.spectralproj.ngs import NGvecs, SpectralProjNG
from arf import ARF


a = ARF(scaling=15)
A, B, X = a.selfadjsystem(p=2)

rad = 1
ctr = 3069
npts = 8
P = SpectralProjNG(X, A.mat, B.mat, rad, ctr, npts, reduce_sym=True)
Y = NGvecs(X, 10, B)
Y.setrandom()
lam, Y, history, Yl = P.feast(Y, hermitian=True, stop_tol=1e-6)
Y.draw()
