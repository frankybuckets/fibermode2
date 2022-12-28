#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:25:45 2022

@author: pv
"""

import numpy as np
from os.path import expanduser, relpath
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

plt.close('all')

main = expanduser('~/local/fiberamp/fiber/microstruct/bragg/notebooks/exact/\
Polymer_Coated/data/')


n = 300
wls = np.linspace(1.4e-6, 2e-6, n+1)
# Set up the figure and subplots
fig, (ax1) = plt.subplots(1, 1, sharex=False, figsize=(28, 14))

styles = [
    {'lw': 1.2, 'msz': 0, 'ls': '-', 'm': '^',
     'c': 'blue', 'label': 'k = 0.000'},

    {'lw': 2, 'msz': 0, 'ls': '-', 'm': '^',
     'c': 'darkorange', 'label': 'k = 0.001'},

    {'lw': 2.5, 'msz': 0, 'ls': '-', 'm': '^',
     'c': 'g', 'label': 'k = 0.005'},

    {'lw': 2.5, 'msz': 0, 'ls': '-', 'm': '^',
     'c': 'red', 'label': 'k = 0.01'},

    {'lw': 2.5, 'msz': 0, 'ls': '--', 'm': '^',
     'c': 'k', 'label': '$N_0$'},
]

materials = [
    'no_loss',
    'low_loss',
    'med_loss',
    'high_loss',
    'N0'
]

# Plot the data
for s, d in zip(materials, styles):
    CL = np.load(relpath(main + s + '_betas.npy'))
    ax1.plot(wls[~np.isnan(CL)], CL[~np.isnan(CL)], ls=d['ls'],
             label=d['label'], linewidth=d['lw'], markersize=d['msz'],
             marker=d['m'], color=d['c'])
# ax1.plot(wls, -N0betas.imag)

# Set Figure and Axes parameters ################################

# Set titles
# fig.suptitle("Kolyadin Fiber: Fundamental Mode Losses \n\
# for Lossy Polymer Coatings",
#              fontsize=30)


# Set axis labels
ax1.set_xlabel("\nWavelength", fontsize=28)
ax1.set_ylabel("CL\n", fontsize=28)

# Set up ticks and grids

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

ax1.xaxis.set_major_locator(MultipleLocator(1e-7))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(1))
ax1.grid(which='major', color='#CCCCCC', linewidth=1.2, linestyle='--')
ax1.grid(which='minor', color='#CCCCCC', linestyle=':')

# # Set log scale on y axes
ax1.set_yscale('log')

# Turn on subplot tool when graphing to allow finer control of spacing
# plt.subplot_tool(fig)

# After fine tuning, these are the values we want (use export from tool)
plt.subplots_adjust(top=0.905,
                    bottom=0.11,
                    left=0.065,
                    right=0.95,
                    hspace=0.2,
                    wspace=0.2)

ax1.set_ylim(1e-6, 4e-2)
ax1.legend(fontsize=25)
# Show figure (needed for running from command line)
plt.show()
