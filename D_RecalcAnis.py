import numpy as n
from numpy import array, in1d, vstack, hstack, dstack, sin, cos
from Utilities import increm_strains, increm_rot, deeq, pair
from Yld043D import PHI
import matplotlib.pyplot as p
import figfun as f
import os, glob, shutil
from sys import argv
from tqdm import trange
from scipy.interpolate import interp1d
    
'''
This script will:
    - Recalc only the anisotropic eq. stns by opening up:
        NewFilterPassingPoints_3med.npy
        NewFilterResults_3med.dat
        PointsInLastWithStrains.npy
'''

# Both npy
# [0-11]Same as AramisBinary
# [12,13,14] e00, e01, e11
# [15,16,17] evm, eH8, eAnis

# NewFilterResults_3med.dat
# [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp

expt = argv[1]
# Directory name and prefix of the npy
pname = 'TTGM-{}_FS19SS6'.format(expt)
key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
alpha = key[ key[:,0] == int(expt), 4]
# [0]Alpha, [1]VM, [2]H8, [3]Anis
alpha_beta = n.genfromtxt('PlaneStn_YldFun.dat', delimiter=',')
beta_h8 = interp1d(alpha_beta[:,0], alpha_beta[:,2]).__call__(alpha)

STF = n.genfromtxt('../{}/STF.dat'.format(pname), delimiter=',')
last = int(STF[-1,0])

sig00 = STF[:,2]/2  # Hoop sts (assumed 1/2 axial)
sig11 = STF[:,2]   # Ax sts
sig01 = STF[:,3]   # Sh sts
sigvm = n.sqrt(sig11**2 + sig00**2 - sig00*sig11 + 3*sig01**2)
s1 = sig00/2 + sig11/2 + n.sqrt(sig00**2 - 2*sig00*sig11 + 4*sig01**2 + sig11**2)/2
s2 = sig00/2 + sig11/2 - n.sqrt(sig00**2 - 2*sig00*sig11 + 4*sig01**2 + sig11**2)/2
# Hosford eq. sts
sigh8 = (((s1-s2)**8+(s2-0)**8+(0-s1)**8)/2)**(1/8)

# Load up the npys
A = n.load('../{0}/IncrementalAnalysis/PointsInLastWithStrains.npy'.format(pname))
Acopy = A[-1].copy()
# Rotations
R = n.load('../{0}/IncrementalAnalysis/PointsInLast_Rotations.npy'.format(pname))
# 
C = n.load('../{0}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(pname))[-1]

oldloc = C[:,15].argmax()
oldmax = C[oldloc, [15,16,17,12,13,14]]
oldmean = C[:,[15,16,17,12,13,14]].mean(axis=1)

# Restore increment!
A[...,12:] = n.vstack((A[[0],:,12:], n.diff(A[...,12:], axis=0)))
# Check correctness via
# n.allclose(A_orig[...,i], A[...,i].cumsum(axis=0))

# An empty 3D array where I'll store each point's de00, de01, de11, and deeqVM, deeqH8
for k in trange(1,last+1):
    q = R[k]
    # sig00t has shape numpts
    sig00t = sig00[k]*cos(q)**2 + 2*sig01[k]*sin(q)*cos(q) + sig11[k]*sin(q)**2
    sig01t =  (sig11[k]-sig00[k])*sin(q)*cos(q) + sig01[k]*(cos(q)**2-sin(q)**2)
    sig11t = sig00[k]*sin(q)**2 - 2*sig01[k]*sin(q)*cos(q) + sig11[k]*cos(q)**2    
    sigAnis = PHI(sig00t, sig11t, sig01t)
    A[k,:,17] = deeq(A[k,:,12:15], 
                     [i for i in (sig00[k], sig01[k], sig11[k], sigAnis)] )    
    A[k,:,12:]+=A[k-1,:,12:] # effectively cumsumming as I go
    # Now append and save to dnew

# Check if I've done it wrong    
assert n.allclose(A[-1,:,:-1], Acopy[:,:-1])
 
# Only keep the passing points
allIDs = pair(A[-1,:,:2])
passing = pair(C[:,:2])
C = C[ passing.argsort() ]
oldloc = C[:,15].argmax()
keeps = n.in1d(allIDs, passing)
D = A.compress(keeps, axis=1)
Dorder = pair(D[-1,:,:2]).argsort()
D = D.take(Dorder, axis=1)
locmax = D[-1,:,15].argmax()
assert n.allclose(C[:,:-1], D[-1,:,:-1])
assert n.allclose(C[oldloc,:-1], D[-1,locmax,:-1])

# The results summary
res = n.genfromtxt('../{0}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(pname), delimiter=',')
header = 'This is the new column filtering\n'
header += 'Max Pt IJ {}, {}\n'.format(*D[-1,locmax,:2].astype(int))
header += '[0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp'
res[:,:6] = D[...,[15,16,17,12,13,14]].mean(axis=1)
res[:,6:12] = D[:, locmax, [15,16,17,12,13,14]]

n.savetxt('../{}/IncrementalAnalysis/NewFilterResults_{}med.dat'.format(pname, 3), 
            fmt='%.6f', delimiter=',', header=header, 
            X=res)
n.save('../{}/IncrementalAnalysis/PointsInLastWithStrains.npy'.format(pname), A)
n.save('../{}/IncrementalAnalysis/NewFilterPassingPoints_{}med.npy'.format(pname, 3),D)
