# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:43:58 2020
"""

import numpy as np
import math
from numpy import linalg as la
import control
from math import factorial
import matplotlib.pyplot as plt
import scipy.special
import matplotlib.ticker as tick
import matplotlib
import networkx as nx
from matplotlib.font_manager import FontProperties
import sys

# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


linestyle_tuple = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),

     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

mylist = ['dotted', 'densely dashed', 'densely dashdotdotted', 'dashed', 'densely dashdotted', 'solid']

sigma = 3
delta = 0.1
T = 20
N=20
data = []
time = list(range(1,T))
data.append([(sigma**2/t)**t for t in time])
plt.plot(time, data[-1], linestyle=linestyle_tuple[mylist[0]])
linecounter = 1
for j in range(1,T):
    timej = list(range(j+1,T))
    data.append([delta**j *(factorial(t)/(factorial(j)*factorial(t-j)))* (sigma**2/(t-1))**(t-1) for t in timej])
    if j <= N/5:
        plt.plot(timej, data[-1], linestyle=linestyle_tuple[mylist[linecounter]])
        linecounter = linecounter + 1

sum =[]
for i in range(T-1):
    temp = data[0][i]
    for k in range(0,i):
        temp += data[k+1][i-k-1]
    sum.append(temp)

plt.plot(time, sum, color='k')#,marker = '*')

plt.xlabel('Iteration $t$')
plt.xticks(np.arange(min(time), max(time)+1, 2.0))
plt.fill_between(time, data[0], sum, facecolor='green', interpolate=True, alpha=0.5)# where=data[0] <= sum,

leg = ['Term %d' % d for d in range(1, int(N/5+2))]
leg[0] = 'Term 1 (Lower-bound)'
leg.append('Sum (Upper-bound)')
leg.append('$M_t^2(A)$')
plt.legend(leg)
plt.grid()
plt.tight_layout()