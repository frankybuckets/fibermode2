
# Geometric Parameters.  Dimensional

p = 6                      # number of sides of polygonal lattice
layers = 6                 # number of layers of lattice
skip = 2                   # number of layers to skip before beginning lattice
pattern = []               # pattern determining microstructure

Λ = 7 * 10**-6                          # separation between layers
r_tube = .5 * .4 * Λ                    # radius of inner tubes
r_fiber = (skip + layers + 1) * Λ       # radius of fiber
r_core = .7 * (Λ * skip - r_tube)       # radius of core region
scale = .5 * (Λ * skip - r_tube)        # scaling factor

# Note: avoid changing scaling parameter.  Changing it moves your eigenvalues

# Physical Parameters

n_tube = 1.48                       # refractive index of tube material
n_clad = 1.45                       # refractive index of cladding material
n_core = n_clad                     # refractive index of core
n_buffer = n_clad                   # refractive index of buffer region
n_outer = n_clad                    # refractive index of outer PML region
n0 = n_clad                         # base refractive index for V function
wavelength = 1.55e-6


# PML Parameters.  Dimensional

t_buffer = 0                   # thickness of buffer region (between R0 and R)
t_outer = 1.5 * 15 * Λ         # thickness of outer region (between R and Rout)
alpha = 5                      # PML factor


# Mesh Parameters. Non-Dimensional

pml_maxh = .5 * r_fiber / scale
buffer_maxh = .1 * r_fiber / scale
tube_maxh = .05 * r_fiber / scale
clad_maxh = .4 * r_fiber / scale
core_maxh = .02 * r_fiber / scale

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
    'n_core': n_core,
    'n_outer': n_outer,
    'n0': n0,
    'wavelength': wavelength,

    't_buffer': t_buffer,
    't_outer': t_outer,
    'alpha': alpha,

    'pml_maxh': pml_maxh,
    'buffer_maxh': buffer_maxh,
    'tube_maxh': tube_maxh,
    'clad_maxh': clad_maxh,
    'core_maxh': core_maxh,
}
