"""
Loading saved ARF object and modes
"""

from fiberamp.fiber.microstruct import loadarfmode

# Example: load the mode computed and saved by arf_poletti_mode.py

a, Y, betas, Zs, p, allprms, _, _ = \
    loadarfmode('outputs/tmp_arf_LP01_mde.npz', 'outputs/tmp_arf_LP01')

Y.draw()
