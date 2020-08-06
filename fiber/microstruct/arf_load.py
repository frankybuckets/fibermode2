"""
Loading saved ARF object and modes
"""

from arf import loadarfmode

a, Y, betas, p, allprm = loadarfmode('outputs/arfLP01_p2_r1')
Y.draw()
