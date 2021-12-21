import numpy as np
# Geometric Parameters.  Dimensional

p = 6                      # number of sides of polygonal lattice
layers = 1                 # number of layers of lattice
skip = 2                   # number of layers to skip before beginning lattice
pattern = []               # pattern determining microstructure

Λ = 7                          # separation between layers
r_tube = .5 * .4 * Λ                    # radius of inner tubes
r_fiber = (skip + layers + .5) * Λ       # radius of fiber
r_core = .7 * (Λ * skip - r_tube)       # radius of core region
scale = 1                               # scaling factor


# Refractive Indices

n_tube = 1.48                       # refractive index of tube material
n_clad = 1.45                       # refractive index of cladding material
n_core = n_clad                     # refractive index of core
n_poly = n_clad                     # refractive index of polymer
n_buffer = n_clad                   # refractive index of buffer region
n_outer = n_clad                    # refractive index of outer PML region
n0 = n_clad                         # base refractive index for V function

# Physical Parameters

wavelength = 1.55
eps0 = 1
mu0 = 1
v0 = 1 / (eps0 * mu0) ** .5
freq = v0 / wavelength
omega = 2 * np.pi * freq

# PML Parameters.  Dimensional

t_poly = 1                   # polymer jacket thickness
t_buffer = 0                 # thickness of buffer region (between R0 and R)
t_outer = .8 * r_fiber       # thickness of outer region (between R and Rout)
alpha = 5                    # PML factor
pml_type = 'radial'
square_buffer = .25

# Mesh Parameters. Non-Dimensional

pml_maxh = .7 * r_fiber / scale
buffer_maxh = .5 * r_fiber / scale
tube_maxh = .5 * r_fiber / scale
clad_maxh = .8 * r_fiber / scale
poly_maxh = .8 * r_fiber / scale
core_maxh = .3 * r_fiber / scale

params = {

    'p': p,
    'layers': layers,
    'skip': skip,
    'pattern': pattern,
    'Λ': Λ,
    'r_tube': r_tube,
    'r_core': r_core,
    'r_fiber': r_fiber,
    'scale': scale,

    'n_buffer': n_buffer,
    'n_tube': n_tube,
    'n_clad': n_clad,
    'n_poly': n_poly,
    'n_core': n_core,
    'n_outer': n_outer,
    'n0': n0,
    'wavelength': wavelength,

    't_buffer': t_buffer,
    't_poly': t_poly,
    't_outer': t_outer,
    'alpha': alpha,
    'pml_type': pml_type,
    'square_buffer': square_buffer,

    'pml_maxh': pml_maxh,
    'buffer_maxh': buffer_maxh,
    'tube_maxh': tube_maxh,
    'clad_maxh': clad_maxh,
    'poly_maxh': poly_maxh,
    'core_maxh': core_maxh,
}
