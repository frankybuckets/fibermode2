"""
Numerically compute guided LP modes of a step-index fiber
"""

from fibermode import StepIndex

fb = StepIndex(fibername='Nufern_Yb')
betas, zsqrs, Y = fb.guidedmodes(p=3)

# name the betas by LP convention and compare with exact betas
n2i, exactbetas = fb.name2indices(betas)

# report
print('#' * 64, '\nRESULTS:', '#' * 55)
print('Computed non-dimensional Z-squared values:\n', zsqrs)
print('LP names:\n', n2i)
print('Computed approximation of physical propagation constants:\n', betas)
print('Exact physical propagation constants:\n', exactbetas)
print('#' * 64)

# save results into a temporary file (all saved files are in "outputs" folder.)
#
#   (See loadmodes.py on how to load modes saved in this file.)
fb.savemodes('my_tmp_output', betas, Y)
