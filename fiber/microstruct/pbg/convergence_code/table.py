#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:10:31 2021.

@author: pv
"""
import numpy as np
import os
from prettytable import PrettyTable

# %%
mode_types = ['A', 'B']
mode = 'LP11'
mode_type = 'A'
r = 1
mode_type_index = mode_types.index(mode_type)

ps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
refs = [0, 1, 2, 3]

folder = '/home/pv/local/fiberamp/fiber/microstruct/pbg/\
outputs/lyr6cr2/' + mode + '/convergence'

if not os.path.isdir(os.path.relpath(folder)):

    raise ValueError("Given folder is not a directory.  Make this directory \
and begin again.")

D = {}

for ref in refs:
    ref_D = {}
    CL = []
    ndofs = []

    for p in ps:
        try:
            filename = 'p' + str(p) + '_refs' + str(ref) + '.npz'
            filepath = os.path.abspath(folder + '/' + filename)
            d = np.load(filepath)
            if mode_type_index + 1 > len(d['z']):
                raise IndexError("Chosen mode does not have this mode type.")
            CL.append(d['CL'][mode_type_index])
            ndofs.append(d['ndofs'])

        except FileNotFoundError:
            CL.append('--')
            ndofs.append('--')

    ref_D['CL'] = CL
    ref_D['ndofs'] = ndofs
    D[ref] = ref_D

name = mode + '-' + mode_type
title = 'Mode: ' + name + ', Refinements: ' + str(r)

table = PrettyTable(title=title, padding_width=5)
fields = ['Refinements', 'Degree', 'Confinement Loss (dB/m)', 'DoFs']

table.add_column(fields[1], ps)
table.add_column(fields[2], D[r]['CL'])
table.add_column(fields[3], D[r]['ndofs'])


print(table)
