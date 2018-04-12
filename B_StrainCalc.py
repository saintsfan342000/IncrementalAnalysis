import numpy as n
from numpy import array, in1d, vstack, hstack, dstack
import matplotlib.pyplot as p
import figfun as f
import os, glob, shutil
from sys import argv
try:
    from tqdm import trange
    myrange=trange
except ImportError:
    myrange=range
    
'''
This script will:
    - Calculate the incremental strains for every point in the npz
    - Append those strains to the incoming data and save into a new npz
'''

expt = argv[1]
# Directory name and prefix of the npy
pname = 'TT2-{}_FS19SS6'.format(expt)

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')

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

dmean = n.genfromtxt('../{}/mean.dat'.format(pname), delimiter=',', usecols=(11))
maxij = n.genfromtxt('../{}/max.dat'.format(pname), delimiter=',', usecols=(11,12))
dmax = n.genfromtxt('../{}/MaxPt.dat'.format(pname), delimiter=',', usecols=(10))

dr = n.genfromtxt('../{}/disp-rot.dat'.format(pname), delimiter=',', usecols=(4,5))
if expt == '9':
    dr = dr[:,0]
    xlab = '$\\delta/\\mathsf{L}$'
else:    
    dr = dr[:,1]
    xlab = '$\\Phi$'

# Increm strains
def increm_strains(A,B):
    '''
    Requires A and B be (nx4), with cols corresponding to F00, F01, F10, F11
    '''
    de00 = (2*((A[:,0] - B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] - B[:,1])*(A[:,2] + B[:,2]))/
            ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
           )
    de01 = ((-(A[:,0] - B[:,0])*(A[:,1] + B[:,1]) + (A[:,0] + B[:,0])*(A[:,1] - B[:,1]) + 
            (A[:,2] - B[:,2])*(A[:,3] + B[:,3]) - (A[:,2] + B[:,2])*(A[:,3] - B[:,3]))/
            ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
           )
    de11 = (2*((A[:,0] + B[:,0])*(A[:,3] - B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] - B[:,2]))/
            ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
           )
    return de00, de01, de11

def deeq(E,sig):
    '''
    E must be  nx3, columns corresponding to de00, de01, de11
    sig must be a list or tuple of (sig00, sig01, sig11, sigeq)
    '''
    return (E[:,0]*sig[0]+E[:,2]*sig[2]-2*E[:,1]*sig[1])/sig[3]

# Load up the npz
d = n.load('../{0}/{0}_PointsInLast.npz'.format(pname))
# An empty 3D array where I'll store each point's de00, de01, de11, and deeqVM, deeqH8
de = n.empty((last+1, d['stage_0'].shape[0], 5))
de[0] = 0
# Empty array for appended strains
dnew = {}
dnew['stage_0'] = n.c_[d['stage_0'], de[0]]
for k in myrange(1,last+1):
    Aname = 'stage_{}'.format(k)
    Bname = 'stage_{}'.format(k-1)
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
    A = d[Aname]
    B = d[Bname]
    # One more dummy check to make sure A and B have the same shape and point ordering
    if not n.array_equal(A[:,:2], B[:,:2]):
        print("Stage {} arrays aren't equal!")
    de[k,:,:3] = n.c_[increm_strains(A[:,8:], B[:,8:])]
    de[k,:,3] = deeq(de[k,:,:3], [i[k] for i in (sig00, sig01, sig11, sigvm)])
    de[k,:,4] = deeq(de[k,:,:3], [i[k] for i in (sig00, sig01, sig11, sigh8)])
    de[k]+=de[k-1] # effectively cumsumming as I go
    # Now append and save to dnew
    dnew[Aname] = n.c_[A, de[k]]
    if not n.array_equal(d[Aname], dnew[Aname][:,:12]):
        raise ValueError('Your arrays arent equivalent!')

d.close()
n.savez_compressed('../{}/{}_PointsInLastWithStrains.npz'.format(pname, pname), **dnew)
    
# Let's see if old-style calc last stage max is in the new datasetmaxloc = n.nonzero( (A[:,0] == maxij[-1,0]) & (A[:,1] == maxij[-1,1]) )[0]
oldmaxloc = n.nonzero( (A[:,0] == maxij[-1,0]) & (A[:,1] == maxij[-1,1]) )[0]
# And calcualte the new maxloc
newmaxloc = n.argmax(de[-1, :, 3])

p.style.use('mysty')
fig1 = p.figure()
ax1 = fig1.add_subplot(111)
for i in de[:,:,3].T:
    p.plot(dr, i, 'C1', alpha=0.1)
p.plot(dr, de[:,:,3].mean(axis=1),'C0', label='Increm. Mean')
p.plot(dr, dmean, 'C2', label='Old Mean')
p.plot(dr, dmax, 'C3', label='Old Max')
if len(oldmaxloc) == 1:
    p.plot(dr, de[:,oldmaxloc,3], 'C4', label='Old Max Pt')
    if oldmaxloc[0] != newmaxloc:
        p.plot(dr, de[:,newmaxloc,3], 'C5', label='New Max Pt')
    else:
        f.eztext(ax1, 'Old pMax pt is same\nas new max pt.', 'ul')
else:
    p.plot(dr, de[:,newmaxloc,3], 'C5', label='New Max Pt')
    f.eztext(ax1, 'Old Max Pt is not\nin the new dataset', 'ul')
p.xlabel(xlab)
p.ylabel('$\\mathsf{e}_\\mathsf{e}$')
ax1.axis(xmin=0)
f.myax(p.gca())
f.ezlegend(p.gca())
p.savefig('../{}/IncrementalAnalysis1.png'.format(pname), dpi=125)

# Epsilon v gamma
fig2 = p.figure()
ax2 = fig2.add_subplot(111)
for gam, eps in zip(-2*de[:,:,1].T, de[:,:,2].T):
    ax2.plot(gam, eps, 'C1', alpha=0.05)
ax2.plot(-2*de[:,:,1].mean(axis=1), de[:,:,2].mean(axis=1), label='Mean', zorder=50)
if len(oldmaxloc) == 1:
    ax2.plot(-2*de[:,oldmaxloc,1], de[:,oldmaxloc,2], label='Old Max Pt')
    if oldmaxloc[0] != newmaxloc:
        ax2.plot(-2*de[:,newmaxloc,1], de[:,newmaxloc,2], label='New Max Pt')
else:
    ax2.plot(-2*de[:,newmaxloc,1], de[:,newmaxloc,2], label='New Max Pt')
ax2.set_xlabel('$\\mathsf{2e}_{\\theta\\mathsf{x}}$')
ax2.set_ylabel('$\\mathsf{e}_\\mathsf{x}$')
ax2.axis(xmin=0, ymin=0)
f.myax(ax2)
f.ezlegend(ax2)
fig2.savefig('../{}/IncrementalAnalysis2.png'.format(pname), dpi=125)