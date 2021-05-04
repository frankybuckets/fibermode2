
# Geometric Parameters.  Dimensional

p = 6
layers = 7
skip = 1
pattern = []
sep = 7 * 10**-6
r = .5 * .4 * sep
S = (skip + layers + 1) * sep
scale = .8 * (sep * skip - r)


# Physical Parameters

n_tube = 1.48
n_clad = 1.45


# PML Parameters.  Dimensional

t_air = 2 * sep
t_outer = 6 * sep
alpha = 5

# Mesh Parameters. Non-Dimensional

pml_maxh = .5 * S / scale
air_maxh = .1 * S / scale
tube_maxh = .05 * S / scale
clad_maxh = .1 * S / scale
core_maxh = .08 * S / scale


params = {

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

    't_air': t_air,
    't_outer': t_outer,
    'alpha': alpha,

    'pml_maxh': pml_maxh,
    'air_maxh': air_maxh,
    'tube_maxh': tube_maxh,
    'clad_maxh': clad_maxh,
    'core_maxh': core_maxh

}
