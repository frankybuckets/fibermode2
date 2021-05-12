
# Geometric Parameters.  Dimensional

p = 6
layers = 1
skip = 0
pattern = []
sep = 7 * 10**-6
r_tube = 2 * sep
r_fiber = 16 * r_tube
r_core = r_tube
scale = r_core


# Physical Parameters

n_air = 1.00027717
n_tube = 1.48
n_clad = 1.45
n_core = n_clad
n_outer = n_clad
n_base = n_clad
wavelength = 1.55e-6


# PML Parameters.  Dimensional

t_air = 0
t_outer = 6 * sep
alpha = 5

# Non-Dimensional radii

R0 = r_core / scale
R = (r_fiber + t_air) / scale
Rout = (r_fiber + t_air + t_outer) / scale
R_fiber = r_fiber / scale

# Mesh Parameters. Non-Dimensional

pml_maxh = .5 * r_fiber / scale
air_maxh = .1 * r_fiber / scale
tube_maxh = .05 * r_fiber / scale
clad_maxh = .4 * r_fiber / scale
core_maxh = .1 * r_fiber / scale

# Refractive index dictionary

n_dict = {'Outer': n_outer,
          'clad': n_clad,
          'tube': n_tube,
          'air': n_air,
          'core': n_core
          }


params = {

    'p': p,
    'layers': layers,
    'skip': skip,
    'pattern': pattern,
    'sep': sep,
    'r_tube': r_tube,
    'r_core': r_core,
    'r_fiber': r_fiber,
    'scale': scale,

    'n_air': n_air,
    'n_tube': n_tube,
    'n_clad': n_clad,
    'n_core': n_core,
    'n_outer': n_outer,
    'n_base': n_base,
    'wavelength': wavelength,

    't_air': t_air,
    't_outer': t_outer,
    'alpha': alpha,

    'R0': R0,
    'R': R,
    'Rout': Rout,
    'R_fiber': R_fiber,

    'pml_maxh': pml_maxh,
    'air_maxh': air_maxh,
    'tube_maxh': tube_maxh,
    'clad_maxh': clad_maxh,
    'core_maxh': core_maxh,
    'n_dict': n_dict
}
