"""
File containing named step-index fiber parameters
"""

from math import sqrt, pi

named_stepindex_fibers = {
    'Nufern_Yb': {
        # ** 25/400 Ytterbium-Doped LMA Double Clad Fiber **
        # From Nufern company Spec sheet provided by Grosek:
        'rcore': 1.25e-5,
        'rclad': 2e-4,
        'wavelen': 1.064e-6,
        'ncore': 1.450971,
        'NA': 0.06
    },
    'Nufern_Tm': {
        # ** (25/400) Thulium-Doped LMA Double Clad Fiber **
        # From Spec sheet provided by Grosek:
        'rcore': 1.25e-5,
        'rclad': 2e-4,
        'wavelen': 2.110e-6,
        'ncore': 1.439994,
        'NA': 0.01
    },
    'corning_smf_28_1': {
        # Single mode fiber from Schermer and Cole paper,
        # as well as BYU specs document from April 2002.
        'rcore': 4.1e-6,
        'rclad': 125e-6,
        'wavelen': 1.320e-6,
        'ncore': sqrt(1.447**2 + 0.117**2),
        'nclad': 1.447,
        'NA': 0.117
    },
    'corning_smf_28_2': {
        # Single mode fiber from Schermer and Cole paper,
        # as well as BYU specs document from April 2002.
        'rcore': 4.1e-6,
        'rclad': 125e-6,
        'wavelen': 1.550e-6,
        'ncore': sqrt(1.440**2 + 0.117**2),
        'nclad': 1.440,
        'NA': 0.117
    },
    'liekki_1': {
        # Multi-mode mode fiber from Schermer and Cole paper,
        # with cladding radius obtained from nLight spec sheet
        # on Liekki passive 25/250DC fibers (slightly different
        # specs than the specified 25/240DC in Schermer and Cole).
        'rcore': 1.25e-5,
        'rclad': 1.25e-4,
        'wavelen': 6.33e-7,
        'ncore': sqrt(1.46**2 + 0.06**2),
        'nclad': 1.46,
        'NA': 0.06
    },
    'liekki_2': {
        # Multi-mode mode fiber from Schermer and Cole paper,
        # with cladding radius obtained from nLight spec sheet
        # on Liekki passive 25/250DC fibers (slightly different
        # specs than the specified 25/240DC in Schermer and Cole).
        'rcore': 1.25e-5,
        'rclad': 1.25e-4,
        'wavelen': 8.30e-7,
        'ks': 2 * pi / 8.30e-7,
        'ncore': sqrt(1.46**2 + 0.06**2),
        'nclad': 1.46,
        'NA': 0.06
    }
}
