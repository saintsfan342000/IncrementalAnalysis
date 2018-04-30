import numpy as n
from numpy import array, in1d, vstack, hstack, dstack
import matplotlib.pyplot as p
p.style.use('mysty')
import figfun as f
import os, glob, shutil
from sys import argv
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter #, spline_filter
from FtfEig import Eig2x2, LEp

'''
This script will:
    - Implement the filtering schemes I identified in Exploratory ipynbs, namely:
        - Only maxima that:
            (1) Are in continous columns with 2t/3 on each side
            (2) Have all 8 neighboring points
            (3) That are reasonably-well behaved in the final minute of the ext
    - Of those that satisfy (1) and (2), I also:
        - Average F of all their neighbors in each stage
        - Compute the mean and max of the maxima based on this technique

'''

def pair(D):
    '''
    Cantor pairing function from wikipedia
    D must be (nx2)
    '''
    if (D.ndim != 2) and (D.shape[1] != 2):
        raise ValueError('Array must be nx2')
    else:
        return (D[:,0] + D[:,1]) * (D[:,0] + D[:,1] + 1)/2 + D[:,1]

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

try:
    expt = argv[1]
except IndexError:
    expt = '24'
# Directory name and prefix of the npy
pname = 'TT2-{}_FS19SS6'.format(expt)

key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
thick = key[ key[:,0] == int(expt), 6 ].ravel()

Xmin, Xmax, Ymin, Ymax = n.genfromtxt('../{}/box_limits.dat'.format(pname),
                                       delimiter=',')
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
maxIJs = n.genfromtxt('../{}/max.dat'.format(pname), delimiter=',', usecols=(11,12))
dmax = n.genfromtxt('../{}/MaxPt.dat'.format(pname), delimiter=',')

d = n.load('../{0}/IncrementalAnalysis/PointsInLastWithStrains.npz'.format(pname))
A = n.empty((last+1,*d['stage_0'].shape))
for k in range(last+1):
    A[k] = d['stage_{}'.format(k)].copy()
d.close()
ID = pair(A[-1,:,:2])

dr = n.genfromtxt('../{}/disp-rot.dat'.format(pname), delimiter=',', usecols=(4,5))
if expt == '18':
    dr = dr[:,0]
    xlab = '$\\delta/\\mathsf{L}$'
    # Swap out shear (13) with hoop (12) strain and divide by -2
    A[:,:,12:14] = A[:,:,13:11:-1]/-2
    dmax[:,9] = -dmax[:,7].copy()
else:    
    dr = dr[:,1]
    xlab = '$\\Phi$'

# Each column's profile
# I only want colums that have continous points with +/- 2/3 a wall thickness from the max
maxij = []
allnbhd = []
fig, a, a = f.make12()
fig.clear()
minht = 2*thick/3
for Iind in n.unique( A[-1,:,0] ).astype(int):
    rng = (A[-1,:,0] == Iind)
    colA = A[-1].compress(rng, axis=0) # a 2D array!
    profmaxloc = colA[:,15].argmax()
    # Shift the maxima to x=0
    shift = colA[profmaxloc,3]
    colA[:,[3,6]]-=shift
    if (colA[:,3].max() < minht) or ((colA[:,3]).min() > -minht):
        continue
    rng = ((colA[:,3]) < minht) & ((colA[:,3]) > -minht)
    Yind = colA[rng,1]
    # And now verify I my column is unbroken within this range
    if len(Yind) != (n.max(Yind) - n.min(Yind) +1):
        continue
    # And now verify that it's neighbors are all in A
    Jind = int(colA[profmaxloc,1])
    condition = True
    nbhd = []
    for i in range(Iind-1, Iind+2):
        for j in range(Jind-1, Jind+2):
            condition &= (pair(n.c_[i,j]) in ID)
            if condition == True:
                loc = n.nonzero( ID == pair(n.c_[i,j]) )[0][0]
                nbhd.append(loc)
    if condition == False:
        continue            
    allnbhd.append(nbhd)
    maxij.extend([Iind, colA[profmaxloc, 1], n.nonzero((A[-1,:,0]==Iind)&(A[-1,:,1]==colA[profmaxloc,1]))[0][0]])
    p.plot(colA[:,3]/thick, colA[:,15], alpha=0.25, label=str(int(Iind)))
    p.plot(colA[profmaxloc,3]/thick, colA[profmaxloc,15], 'r.', zorder=1000)
    p.plot(colA[[0,-1],3]/thick, colA[[0,-1],15],'ko',zorder=1000);

p.axvline(-2/3)    
p.axvline(2/3)    
p.xlabel('y$_{\\mathsf{o}}$/t$_{\\mathsf{o}}$')
p.ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}}$')  
f.myax(p.gca())  
fig.savefig('../{}/IncrementalAnalysis/IncrementalAnalysis_Profile.png'.format(pname), dpi=125)
p.close(fig)
maxij = n.array(maxij).reshape(-1,3).astype(int)
maxlocs = maxij[:,2]
allnbhd = n.array(allnbhd).reshape(-1,9).astype(int)

##  Do the F averaging for the maxima
F = n.empty((A.shape[0],maxij[:,2].shape[0],4))*n.nan
for k,r in enumerate(allnbhd[:]):
    for w,z in enumerate(A[:,:,8:12].take(r, axis=1)):
        F[w,k] = z.mean(axis=0)
# And now calculate the strains for these averaged F maxima
# VM, H8, e00, e01, e11
de = n.empty((last+1, F.shape[1] , 5))
de[0] = 0
for k in range(1,F.shape[0]):
    de[k,:,2:] = n.c_[increm_strains(F[k], F[k-1])]
    de[k,:,0] = deeq(de[k,:,2:], [i[k] for i in (sig00, sig01, sig11, sigvm)])
    de[k,:,1] = deeq(de[k,:,2:], [i[k] for i in (sig00, sig01, sig11, sigh8)])
    de[k]+=de[k-1] # effectively cumsumming as I go

# Construct and save the AvgNbhdPassingPts
B = A.take(maxij[:,2], axis=1)
B[:,:,8:12] = F
B[:,:,12:] = de
# last+1, npts, 5
loc = de[-1, :, 0].argmax()
# last+1 by npts
LE = LEp(*(F[:,:,i] for i in range(4)))
LE[0] = 0
loc2 = LE[-1,:].argmax()
n.save('../{0}/IncrementalAnalysis/AvgNbhdPassingPts.npy'.format(pname), B)
header = 'This is the new column filtering, taking a F Nbhd avg\n'
header += '[0-4]Mean VM-H8-de00-01-00, [5-9]Max VM-H8-de00-01-00, [10-11]Mean, max Classic LEp'
n.savetxt('../{0}/IncrementalAnalysis/AvgNbhdResults.dat'.format(pname), 
            fmt='%.6f', delimiter=',', header=header,
            X=n.c_[de.mean(axis=1), de[:, loc, : ], LE.mean(axis=1), LE[:, loc2]] )

# Now further donwselect by filtering the eps-gamma path
fig, ax1, ax2 = f.make12()
start = n.nonzero( STF[:,1] >= STF[-1,1] - 120 )[0]
winlen = len(start)-(len(start)%2+0)
winlen = int(winlen/2) - (int(winlen/2)%2-1)
ax1.plot(-2*A[start[0]:,maxlocs, 13].mean(axis=1), 
         A[start[0]:,maxlocs,14].mean(axis=1),'k',lw=2, zorder=30000, label='Incr. Mean')
ax1.set_title('Epsilon vs Gamma for Max Pts')
#ax1.axis(xmin=0)
ax2.plot(dr, A[:,maxlocs,15].mean(axis=1),'k',lw=2, zorder=30000, label='Incr. Mean')
ax2.set_title('Eeq vs Rot for Max Pts')
ax2.axis(xmin=0)
keeps = n.ones_like(maxlocs, dtype=bool)
for k,loc in enumerate(maxlocs[:]):
    l = []
    # Savgol filter to the last minute of each one
    B = A.take(start, axis=0)
    y = savgol_filter(B[:,loc,14], winlen, 2)
    x = savgol_filter(B[:,loc,13], winlen, 2)
    # Interpolate the filtered y-values using the real x-values
    y_int = interp1d(x,y, fill_value='extrapolate').__call__(B[:,loc,13])
    err = ((y_int-y)**2)**.5
    mn, md, std = err.mean(), n.median(err), err.std()
    #print(len(err), (err>(md+4*std)).sum())
    if n.any(err>(md+5*std)):
        keeps[k] = False
        continue
    l.extend( ax1.plot(-2*A[start[0]:,loc, 13], A[start[0]:,loc,14], alpha=0.4 ) )
   
    ax1.plot(-2*x,y, color=l[-1].get_color())
    ax2.plot(dr, A[:,loc,15], alpha=0.4)
f.eztext(ax1,'{} total\n{} rejected'.format(len(maxlocs), (~keeps).sum()), fontsize=20);
ax1.set_xlabel('2e$_{\\theta\\mathsf{x}}$')
ax1.set_ylabel('e$_\\mathsf{xx}$')
ax2.set_xlabel('$\\Phi$')
ax2.set_ylabel('e$_\\mathsf{eq}$')
f.myax(ax1)
f.myax(ax2)
fig.savefig('../{}/IncrementalAnalysis/IncrementalAnalysis_FilterPerformance.png'.format(pname), dpi=125)
p.close(fig)
# And now further downselect passing
maxij = maxij.compress(keeps, axis=0)
maxlocs = maxij[:,2]

fig, ax1, ax2 = f.make12()
ax1.plot(-2*A[:,maxlocs, 13].mean(axis=1), A[:,maxlocs,14].mean(axis=1),'k',lw=2, zorder=30000)
ax1.set_title('Epsilon vs Gamma for Max Pts')
ax1.axis(xmin=0)
ax2.plot(dr, A[:,maxlocs,15].mean(axis=1),'k',lw=2, zorder=30000)
ax2.set_title('Eeq vs Rot for Max Pts')
ax2.axis(xmin=0)
for k,loc in enumerate(maxlocs):
    ax1.plot(-2*A[:,loc, 13], A[:,loc,14], alpha=0.4) 
    ax2.plot(dr, A[:,loc,15], alpha=0.4) 

ax1.set_xlabel('2e$_{\\theta\\mathsf{x}}$')
ax1.set_ylabel('e$_\\mathsf{xx}$')
ax2.set_xlabel('$\\Phi$')
ax2.set_ylabel('e$_\\mathsf{eq}$')
f.eztext(ax2, 'New/Old Max = {:.3f}/{:.3f}\nNew/Old Mean = {:.3f}/{:.3f}'.format(
                A[-1,maxlocs,15].max(), dmax[-1,-1], A[-1,maxlocs,15].mean(), dmean[-1]))

# Plot old data if it's on there
rng = pair(maxIJs[-1][None]) == ID
tx = ''
if rng.sum() == 1:
    loc = n.nonzero(  rng  )[0][0]
    ax2.plot(dr,A[:,loc,15],'r', label='OldMaxPt/Incr')
    ax1.plot(-2*A[:,loc,13], A[:,loc,14], 'r', label='OldMaxPt/Incr')
    tx += 'OldMaxPt in every stage\n'
    if not(pair(maxIJs[-1][None]) in pair(maxij[:,:2])):
        tx += 'But did not pass new filtering'
else:
    tx += "OldMadPt not in\nevery stage!"
ax1.plot(-dmax[:,9], dmax[:,8],'b--', label='OldMaxPT/Old')
f.eztext(ax1,tx)
f.ezlegend(ax1)
f.myax(ax1)
f.myax(ax2)
p.savefig('../{}/IncrementalAnalysis/IncrementalAnalysis_PassingPaths.png'.format(pname), dpi=125)

# [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
# [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) 
# [12,13,14,15,16] e00, e01, e11, eeqVM, eeqH8
p.figure()
p.tricontourf(A[-1,:,2],A[-1,:,3],A[-1,:,15],256)
for k,loc in enumerate(maxlocs):
    p.plot(A[-1,loc,2], A[-1,loc,3],marker='${}$'.format(k+1), color='k', ms=10)
    p.title('eeq contour with max points identified')
p.savefig('../{}/IncrementalAnalysis/IncrementalAnalysis_Contour.png'.format(pname), dpi=125)
p.close()
    
p.figure()
p.hist(A[-1,:,15].take(maxlocs))
p.title('Histogram of Top {} Points Failure Strain'.format(maxij.shape[0]))
p.savefig('../{}/IncrementalAnalysis/IncrementalAnalysis_Hist.png'.format(pname), dpi=125)
p.close()

header = "[0]AramX, [1]AramI, [2]Row in d['stage_n']"
n.save('../{0}/IncrementalAnalysis/NewFilterPassingPoints.npy'.format(pname),
        A.take(maxij[:,2], axis=1))

F = A.take(maxlocs, axis=1)[:,:,8:12]
A = A.take(maxlocs, axis=1).take([15,16,12,13,14], axis=2)
loc = A[-1, :, 0].argmax()
# last+1 by npts
LE = LEp(*(F[:,:,i] for i in range(4)))
LE[0] = 0
loc2 = LE[-1,:].argmax()
header = 'This is the new column filtering\n'
header += '[0-4]Mean VM-H8-de00-01-00, [5-9]Max VM-H8-de00-01-00, [10-11]Mean, max Classic LEp'
n.savetxt('../{0}/IncrementalAnalysis/NewFilterResults.dat'.format(pname), 
            fmt='%.6f', delimiter=',', header=header, 
            X=n.c_[A.mean(axis=1), A[:, loc, : ], LE.mean(axis=1), LE[:,loc2 ]])
