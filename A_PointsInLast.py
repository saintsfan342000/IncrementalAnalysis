import numpy as n
from numpy import array, in1d, vstack, hstack, dstack
from Utilities import pair
import matplotlib.pyplot as p
import os, glob, shutil
from sys import argv
try:
    from tqdm import trange
    myrange=trange
except ImportError:
    myrange=range

'''

This script will:
    - Open up the npy, starting with the last stage
    - Reduce it to the size of the box
    - ID the points that exist in EVERY stage
    - Save those points
    Since I am going backwards I won't be able to calculate the incremental strain
    
    Note that, a Fortunate consequence here is that the order of points in each stage file
    is identical.  So I can vectorize future incremental calculations.
'''

expt = argv[1]
# Directory name and prefix of the npy
pname = 'TT2-{}_FS19SS6'.format(expt)

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
thick = key[ key[:,0] == int(expt), 6 ].ravel()
# Expand the box limits a bit
box_lims = n.genfromtxt('../{}/box_limits.dat'.format(pname),
                                       delimiter=',')
Xmin, Xmax, Ymin, Ymax = box_lims
Ymin -= thick/2
Ymax += thick/2
                                      
last = n.genfromtxt('../{}/STF.dat'.format(pname), delimiter=',', dtype=int)[-1,0]
                                       
# Empty dict ... I should really use and HDF5 now
d = {}
for k in myrange(last,-1,-1):
    dname = 'stage_{}'.format(k)
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
    A = n.load('../{}/AramisBinary/{}_{:.0f}.npy'.format(pname,pname,k))
    if k == last:
        A = A[ (A[:,2]>=Xmin) & (A[:,2]<=Xmax) & (A[:,3]>=Ymin) & (A[:,3]<=Ymax), :]        
        d[dname] = A
        ij_last = pair(A[:,:2])
        # Initialize to all True
        pts_in_all = in1d(ij_last, ij_last)
    else:
        ij = pair(A[:,:2])
        # The last points that are also in all of the next stages
        pts_in_all &= in1d(ij_last, ij)
        # The k stage points that are in last (just reduce the size)
        pts_in_last = in1d(ij, ij_last)
        d[dname] = A[pts_in_last]
                
ij_in_all = ij_last[pts_in_all]
# Now I can loop back though the dict and get rid of the points that aren't 
D = n.empty((last+1, ij_in_all.shape[0], A.shape[1]))
for k in range(last,-1,-1):
    dname = 'stage_{}'.format(k)
    A = d[dname].copy()
    ij = pair(A[:,:2])
    keeps = in1d(ij, ij_in_all)
    #d[dname] = A[keeps]
    D[k] = A[keeps]
    # Here's an inefficient little logic check for myself
    if (k<last):
        ij_prev = pair(D[k+1,:,:2])
        ij = pair(D[k,:,:2])
        if not n.array_equal(ij,ij_prev):
            raise ArithmeticError("You're logic is bad, ya jerk!")
            
# Now save the 3D array: [last+1, numpts, 12
#n.savez_compressed('../{0}/IncrementalAnalysis/PointsInLast.npz'.format(pname), **d)
n.save('../{}/IncrementalAnalysis/PointsInLast.npy'.format(pname), D)
