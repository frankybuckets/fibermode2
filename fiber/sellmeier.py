"""
Sellmeier formula for refractive index variations with wavelength
https://en.wikipedia.org/wiki/Sellmeier_equation
"""
import numpy as np


def index(lam, material='FusedSilica'):
    """
    INPUT:
       Wavelength "lam" in meters.
       Name of "material" for which Sellmeier coefficients are implemented.
    OUTPUT:
       Refractive index at the input wavelength.
    """

    if material == 'FusedSilica':

        # properties for room temperature (20 C)

        B = np.array([0.6961663, 0.4079426, 0.8974794])
        L = np.array([0.0684043, 0.1162414, 9.0896161])

    else:
        raise NotImplementedError('Implement Sellmeier for %s' % material)

    lam *= 1e6
    return np.sqrt(1 + np.sum((B*lam**2) / (lam**2 - L**2)))
