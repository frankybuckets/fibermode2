
# Geometric Parameters.  Dimensional

p = 6
layers = 3
skip = 2
# pattern    = [[1,1,1,1,1,1,1,1,1,1,1,1],\
#     [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0], \
#     [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1]]
pattern = None
sep = 7 * 10**-6
r = .5 * .4 * sep
S = (skip + layers + 2) * sep
scale = S / 2


# Physical Parameters

n_tube = 1.48
n_clad = 1.45
n0 = 1.00027717
wavelength = 1.8e-6


# PML Parameters.  Dimensional

t_air = 2 * sep
t_outer = 6 * sep
alpha = 5

# Mesh Parameters. Non-Dimensional

pml_maxh = .5
air_maxh = .1
tube_maxh = .05
clad_maxh = .1


fiber_param_dict = {

    'p': p,
    'layers': layers,
    'skip': skip,
    'pattern': pattern,
    'sep': sep,
    'r': r,
    'S': S,
    'scale': scale,

    'n_tube': n_tube,
    'n_clad': n_clad,
    'n0': n0,
    'wavelength': wavelength,

    't_air': t_air,
    't_outer': t_outer,
    'alpha': alpha,

    'pml_maxh': pml_maxh,
    'air_maxh': air_maxh,
    'tube_maxh': tube_maxh,
    'clad_maxh': clad_maxh,

}
