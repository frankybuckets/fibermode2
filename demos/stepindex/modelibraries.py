"""Make a library of mode output files for a list of predefined
fibers, using both the numerical (FEAST) eigensolver and the
interpolated semianalytical expressions from StepIndexExact. """

from fibermode import StepIndex

fibnames = ['Nufern_Yb', 'Nufern_Tm']
for fn, maxp in zip(fibnames, [5, 5]):
    fb = StepIndex(fn)
    fb.makeguidedmodelibrary(maxp=maxp, nspan=50)
    fb.makeguidedmodelibrary(maxp=maxp, interp=True)
