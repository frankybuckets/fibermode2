#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:10:31 2021.

@author: pv
"""
import netgen.gui
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params
from fiberamp.fiber.microstruct.pbg import PBG
import numpy as np
import os
import ngsolve as ng
from prettytable import PrettyTable

# %%

table = PrettyTable(padding_width=5)
headers = ['Degree', 'DoFs', 'Confinement Loss (dB/m)']
# table.field_names = headers

ps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
refs = [0]

folder = '/home/pv/local/fiberamp/fiber/microstruct/pbg/\
outputs/lyr6cr2/convergence_studies'

if not os.path.isdir(os.path.relpath(folder)):

    raise ValueError("Given folder is not a directory.  Make this directory \
and begin again.")

all_data = []

for ref in refs:
    data = []
    for p in ps:
        filename = 'p' + str(p) + '_refs' + str(ref) + '.npz'
        filepath = os.path.abspath(folder + '/' + filename)
        d = np.load(filepath)
        data.append(d['CL'][0])
    all_data.append(data)

table.add_column(headers[0], ps)
table.add_column(headers[2], all_data[0])


# %%
center = 1.242933-2.471929e-09j
radius = .01
p = 2

A = PBG(params)
ng.Draw(A.mesh)


# %%

print(A.check_ndof(3, 3))
ng.Draw(A.mesh)

# %%

dofs = []
for p in ps:
    _, X = A.polypmlsystem(p=p, alpha=A.alpha)
    dofs.append(X.ndof)

# %%

table.add_column(headers[1], dofs)
print(table)
