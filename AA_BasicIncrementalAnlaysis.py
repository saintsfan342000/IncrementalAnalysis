import numpy as n
from numpy import array, nanmean, nanstd, sqrt
from numpy import hstack, vstack, dstack
n.set_printoptions(linewidth=300,formatter={'float_kind':lambda x:'{:g}'.format(x)})
from Utilities import increm_strains, deeq, pair
import matplotlib.pyplot as p
import figfun as f
import os
from tqdm import tqdm, trange
from sys import argv
from scipy.io import savemat, loadmat


expt = argv[1]
proj = 'TT2-{}_FS19SS6'.format(expt)
print('\n')
print(proj)

expt = int( proj.split('_')[0].split('-')[1])
FS = int( proj.split('_')[1].split('SS')[0].split('S')[1] )
SS = int( proj.split('_')[1].split('SS')[1] )
arampath = '../{}/AramisBinary'.format(proj)
prefix = '{}_'.format(proj)
savepath = '../{}/IncrementalAnalysis'.format(proj)

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
alpha = key[ key[:,0] == int(expt), 3 ]

# [0]Stg [1]Time [2]AxSts [3]ShSts [4]AxForce [5]Torque [6]MTS Disp [7]MTS Rot
STF = n.genfromtxt('../{}/STF.dat'.format(proj), delimiter=',')
last = int(STF[-1,0])
maxi, maxj = n.genfromtxt('../{}/max.dat'.format(proj), dtype=int,
                        delimiter=',', usecols=(-2,-1), unpack=True)
Xmin,Xmax,Ymin,Ymax = n.genfromtxt('../{}/box_limits.dat'.format(proj), delimiter=',')
if expt == '18':
    dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,4]
    xlab = '$\\delta/\\mathsf{L}$'
else:
    dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,5]
    xlab = '$\\Phi$'

# Point to Point, *only* passing
deeq1 = n.empty((last+1, 5))
# Averaging F  for passing points
deeq2 = n.empty((last+1,5))
# Point to point for passing points
deeq3 = n.empty((last+1,5))
# Max point from last stage traced all the way back
deeq4 = n.empty((last+1,5))
# Max trace back with average of neighbors
deeq6 = n.empty((last+1,5))
# Max point each stage
deeq5 = n.empty((last+1,5))

d = loadmat('../{0}/{0}_PassingPts.mat'.format(proj))
for k in trange(1,last+1):
    
    #if k%25 == 0:  print(k, end=',', flush=True)

    sig00 = STF[k,2]/2  # Hoop sts (assumed 1/2 axial)
    sig11 = STF[k,2]   # Ax sts
    sig01 = STF[k,3]   # Sh sts
    # Mises equivalent sts
    sigvm = n.sqrt(sig11**2 + sig00**2 - sig00*sig11 + 3*sig01**2)
    # Principle stresses
    s1 = sig00/2 + sig11/2 + sqrt(sig00**2 - 2*sig00*sig11 + 4*sig01**2 + sig11**2)/2
    s2 = sig00/2 + sig11/2 - sqrt(sig00**2 - 2*sig00*sig11 + 4*sig01**2 + sig11**2)/2
    # Hosford eq. sts
    sigh8 = (((s1-s2)**8+(s2-0)**8+(0-s1)**8)/2)**(1/8)
    
    # Loading the "passing points" rather than the whole field means I don't have to have
    # all that code used to filter based on eps/gamm ratio
    A = d['stage_{}'.format(k)]
    
    if k == 1:
        B = n.zeros_like(A)
        B[:,2], B[:,-1] = 1, 1
        B[:,:2] = A[:,:2].copy()
   
    ##  deeq2: Average F for all passing!
    if k == 1:
        B00t, B11t = 1, 1
        B01t, B10t = 0, 0
    A00t, A01t, A10t, A11t = [i.mean() for i in A[:,2:].T]
    de00, de01, de11 = increm_strains(A00t,A01t,A10t,A11t,B00t,B01t,B10t,B11t)
    deeq2[k,0] = (sig00*de00 + sig11*de11 - 2*sig01*de01)/sigvm    
    deeq2[k,1] = (sig00*de00 + sig11*de11 - 2*sig01*de01)/sigh8        
    deeq2[k,2:] = n.c_[de00, de01, de11]
    
    # For next stage, keep A and assign it to B
    B00t, B01t, B10t, B11t = A00t, A01t, A10t, A11t
    
    # deeq3:  Point to Point comparison between passing vals
    InB = n.ones(len(A[:,0]), dtype=bool)
    Btemp = n.empty_like(A)*n.nan
    for w, (arX, arY) in enumerate(zip(A[:,0], A[:,1])):
        Brow = B[ (B[:,0]==arX) & (B[:,1]==arY) ].ravel()
        if Brow.shape[0] == 0:
            # Then Brow is empty and those aramX,Y points from A aren't in B
            InB[w] = False
        else:
            Btemp[w] = Brow

    de00, de01, de11 = increm_strains(*[i.compress(InB) 
                                        for i in ( 
                                        (*A[:,2:].T, *Btemp[:,2:].T))])
    deeq3[k,0] = deeq(de00, de01, de11, sig00, sig01, sig11, sigvm)
    deeq3[k,1] = deeq(de00, de01, de11, sig00, sig01, sig11, sigh8)
    deeq3[k,2:] = n.c_[de00, de01, de11]

    ## deeq1:  P2P but only considering passing in A and B
    Bpass = d['stage_{}'.format(k-1)]
    InB = n.ones(len(A[:,0]), dtype=bool)
    Btemp = n.empty_like(A)*n.nan
    for w, (arX, arY) in enumerate(zip(A[:,0], A[:,1])):
        Brow = Bpass[ (Bpass[:,0]==arX) & (Bpass[:,1]==arY) ].ravel()
        if Brow.shape[0] == 0:
            # Then Brow is empty and those aramX,Y points from A aren't in B
            InB[w] = False
        else:
            Btemp[w] = Brow
    
    if k == 1:
        Btemp[:,2:] = 1, 0, 0, 1
        InB[:] = True
    
    de00, de01, de11 = increm_strains(*[i.compress(InB) 
                                        for i in ( 
                                        (*A[:,2:].T, *Btemp[:,2:].T))])
    deeq1[k,0] = deeq(de00, de01, de11, sig00, sig01, sig11, sigvm)
    deeq1[k,1] = deeq(de00, de01, de11, sig00, sig01, sig11, sigh8)
    deeq1[k,2:] = n.c_[de00, de01, de11]


    ## deeq4:  Max Point traced all the way back
    wholeA = n.load('../{0}/AramisBinary/{0}_{1}.npy'.format(proj,k)).take([0,1,8,9,10,11], axis=1)
    maxptA = wholeA[ (wholeA[:,0] == maxi[-1]) & (wholeA[:,1] == maxj[-1]) ].ravel()
    if k == 1:
        maxptB = n.array([0,0,1,0,0,1])
    if (len(maxptA) == 0) or (len(maxptB) == 0):
        deeq4[k] = 0
    else:
        de00, de01, de11 = increm_strains(*(*maxptA[2:], *maxptB[2:]))
        deeq4[k,0] = deeq(de00, de01, de11, sig00, sig01, sig11, sigvm)
        deeq4[k,1] = deeq(de00, de01, de11, sig00, sig01, sig11, sigh8)
        deeq4[k,2:] = n.c_[de00, de01, de11]
    maxptB = maxptA.copy()
        
    ## deeq6:  Max point traced back averaging F with its neighbors
    ID = pair(wholeA[:,:2])
    nbhd = []
    for i in range(maxi[-1]-1, maxi[-1]+2):
        for j in range(maxj[-1]-1, maxj[-1]+2):
            if (pair(n.c_[i,j]) in ID):
                loc = n.nonzero( ID == pair(n.c_[i,j]) )[0][0]
                nbhd.append(loc)
    AFavg = wholeA.take(nbhd, axis=0).mean(axis=0)
    if k == 1:
        BFavg = n.array([0,0,1,0,0,1])
    if (len(AFavg) == 0) or (len(BFavg) == 0):
        deeq6[k] = 0
    else:
        de00, de01, de11 = increm_strains(*(*AFavg[2:], *BFavg[2:]))
        deeq6[k,0] = deeq(de00, de01, de11, sig00, sig01, sig11, sigvm)
        deeq6[k,1] = deeq(de00, de01, de11, sig00, sig01, sig11, sigh8)
        deeq6[k,2:] = n.c_[de00, de01, de11]    
    BFavg = AFavg.copy()
        
    # deeq5:  Max point each stage
    maxptAk = A[ (A[:,0] == maxi[k]) & (A[:,1] == maxj[k]) ].ravel()
    if k == 1:
        maxptBk = n.array([0,0,1,0,0,1])
    if (len(maxptAk) == 0) or (len(maxptBk) == 0):
        deeq5[k] = 0
    else:
        de00, de01, de11 = increm_strains(*(*maxptAk[2:], *maxptBk[2:]))
        deeq5[k,0] = deeq(de00, de01, de11, sig00, sig01, sig11, sigvm)
        deeq5[k,1] = deeq(de00, de01, de11, sig00, sig01, sig11, sigh8)
        deeq5[k,2:] = n.c_[de00, de01, de11]
    maxptBk = maxptAk.copy()

    B = wholeA.copy()
        
deeq2[0] = 0
deeq3[0] = 0
deeq4[0] = 0
deeq5[0] = 0
deeq6[0] = 0

for i in [deeq2, deeq3, deeq4, deeq5, deeq6, deeq1]:
    i[ n.any(n.isnan(i), axis=1) ] = 0

headerline = ('[0-4]AvgF-Passing-VM-H8-de00-01-11, [5-9]PassingP2P-VM-H8, [10-14]MaxPtTracedBack-VM-H8,' + 
                '[15-19]MaxPtEachStage-VM-H8, [20-24]MaxPtTrace/NbhdFavg, [25-29]')
X = n.c_[deeq2.cumsum(axis=0), deeq3.cumsum(axis=0),
         deeq4.cumsum(axis=0), deeq5.cumsum(axis=0), deeq6.cumsum(axis=0), deeq1.cumsum(axis=0)]
fname='../{0}/IncrementalAnalysis/OldFilteringResults.dat'.format(proj)
n.savetxt(fname, X=X, header=headerline,
        fmt = '%.6f', delimiter=', ')


p.style.use('mysty-sub')        
labs = ['Avg_AllPass', 'Avg_P2P', 'LastMax_TraceBk', 'Max_EachStgP2P','LastMax_Trace_Nbhd', 'Avg_P2P_PassOnly']
alpha = [1,1,1,.5,1,1]
ls = ['-', '--', '-', '-', '--', '-', '--']

dmean = n.genfromtxt('../{}/mean.dat'.format(proj))[:,-1]
dmax = n.genfromtxt('../{}/MaxPt.dat'.format(proj))[:,10]

fig, ax1, ax2 = f.make21()
for i in [0,1,5]:
    ax1.plot(dr,X[:,i*5], label=labs[i], alpha=alpha[i], ls=ls[i])
ax1.plot(dr, dmean, 'k--', label='Old')
ax1.set_xlabel(xlab)
ax1.set_ylabel('e$_\\mathsf{e}$')
f.ezlegend(ax1, loc=2, title='Mean Values')
ax1.axis(xmin=0,ymin=0)
f.myax(ax1)

for i in [2,3,4]:
    ax2.plot(dr,X[:,i*5], label=labs[i], alpha=alpha[i], ls=ls[i])
ax2.plot(dr, dmax, 'k--', label='Old')
ax2.set_xlabel(xlab)
ax2.set_ylabel('e$_\\mathsf{e}$')
f.ezlegend(ax2, loc=2, title='Max Values')
ax2.axis(xmin=0,ymin=0)
f.myax(ax2)

p.savefig('../{0}/IncrementalAnalysis/OldFilteringResults.png'.format(proj), dpi=125)
p.close()
