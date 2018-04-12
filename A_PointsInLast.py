import numpy as n
from numpy import array, in1d, vstack, hstack, dstack
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
pname = 'TTGM-{}_FS19SS6'.format(expt)

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
thick = key[ key[:,0] == int(expt), 6 ].ravel()
# Expand the box limits a bit
box_lims = n.genfromtxt('../{}/box_limits.dat'.format(pname),
                                       delimiter=',')
Xmin, Xmax, Ymin, Ymax = box_lims
Ymin -= thick/2
Ymax += thick/2
                                      
last = n.genfromtxt('../{}/STF.dat'.format(pname), delimiter=',', dtype=int)[-1,0]
                                       
def pair(D):
    '''
    Cantor pairing function from wikipedia
    D must be (nx2)
    '''
    if (D.ndim != 2) and (D.shape[1] != 2):
        raise ValueError('Array must be nx2')
    else:
        return (D[:,0] + D[:,1]) * (D[:,0] + D[:,1] + 1)/2 + D[:,1]
                                       
# Empty dict ... I should really use and HDF5 now
d = {}
for k in trange(last,-1,-1):
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
shape = []
for k in range(last,-1,-1):
    dname = 'stage_{}'.format(k)
    A = d[dname].copy()
    ij = pair(A[:,:2])
    keeps = in1d(ij, ij_in_all)
    d[dname] = A[keeps]
    shape.append([*(d[dname].shape)])
    # Here's an inefficient little logic check for myself
    if (k<last):
        dnameprev = 'stage_{}'.format(k+1)
        ij_prev = pair(d[dnameprev][:,:2])
        ij = pair(d[dname][:,:2])
        if (shape[-1][0] != shape[-2][0]) or not n.all(ij==ij_prev):
            raise ArithmeticError("You're logic is bad, ya jerk!")
            
# Now save the dictionary as a compressed numpy array (to save a bit of space)
# Doesn't appear to take any more time to load up whether it's compressed
# Takes a few seconds to save though
n.savez_compressed('../{}/{}_PointsInLast.npz'.format(pname, pname), **d)