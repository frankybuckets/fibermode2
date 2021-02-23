"""
Loading saved ARF object and modes
"""

from arf import loadarfmode

# Run this after computing & saving a mode, say by arf_mode.py

a, Y, betas, Zs, p, allprms, _, _ = \
    loadarfmode('outputs/arfLP02_p2_mde.npz', 'outputs/arfLP02_p2')

Y.draw()
