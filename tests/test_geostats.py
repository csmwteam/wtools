import sys
sys.path.append('../')

from wtools.geostats import *
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# QUESTION 1

new_cal2 = np.loadtxt('new_calibration2.dat', dtype = float)
new_cal_clay = new_cal2[:,2]
new_cal_res = new_cal2[:,5]

fit_coeff = np.polyfit(new_cal_res, new_cal_clay, 1)
res_range = np.linspace(10, 250, 500)
fit_val = np.polyval(fit_coeff, res_range)


plt.scatter(new_cal_res, new_cal_clay, label='Observed')
plt.semilogx(res_range, fit_val, label='Trend')
plt.legend()
plt.title('Calibration Data with Fit Line')
plt.xlabel('Log Resistivity (Ohm-m)')
plt.ylabel('Clay Fraction')
plt.show()


###############################################################################
# QUESTION 2

# Read in data file
n78 = np.loadtxt('new_78_860.dat', dtype=float)
wlog = n78[:,1] == 4

gspecs = GridSpec( n=len(wlog),
                   min=0,
                   sz=1,
                   nnodes=int(0.5*len(wlog)) )

outStruct, outNpairs = raster2structgrid(wlog, gspecs);

plt.plot(outStruct)
###############################################################################
# QUESTION 3


###############################################################################
# QUESTION 4
