"""
Loading saved ARF object and modes
"""

from arf import loadarfmode

# Run this after the arfLP01_p2_r1 output file is created
# by arf_mode.py

a, Y, betas, p, allprm = loadarfmode('outputs/arfLP01_p2_r1')
Y.draw()
