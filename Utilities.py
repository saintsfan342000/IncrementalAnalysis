import numpy as n
import matplotlib.pyplot as p
import figfun as f
from sys import argv
from FtfEig import LEp
from pandas import read_excel

def PlotOldMaxTraceBack(expt):
    '''
    Plots the STK16 max point eeq vs rot and its increm. ex vs 2eqx
    
    '''
    p.style.use('mysty')
    fig, ax = p.subplots()
    labs = ['Avg_AllPass', 'Avg_P2P', 'LastMax_TraceBk', 'Max_EachStgP2P','LastMax_Trace_Nbhd', 'Avg_P2P_PassOnly']
    alpha = [1,1,1,.5,1,1]
    ls = ['-', '--', '-', '-', '--', '-', '--']
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    dmean = n.genfromtxt('../{}/mean.dat'.format(proj))[:,-1]
    dmax = n.genfromtxt('../{}/MaxPt.dat'.format(proj), delimiter=',')[:,10]
    # [0-4]AvgF-Passing-VM-H8-de00-01-11, [5-9]PassingP2P-VM-H8, [10-14]MaxPtTracedBack-VM-H8,
    # [15-19]MaxPtEachStage-VM-H8, [20-24]MaxPtTrace/NbhdFavg, [25-29]
    X = n.genfromtxt('../{}/IncrementalAnalysis/OldFilteringResults.dat'.format(proj), delimiter=',')
    if expt in ['18',18]:
        dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,4]
        xlab = '$\\delta/\\mathsf{L}$'
    else:
        dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,5]
        xlab = '$\\Phi$'  
    for i in [2]:
        ax.plot(dr,X[:,i*5], label='Incremental\nTracedBack')
    ax.plot(dr, dmax, label='STK16')
    ax.set_xlabel(xlab)
    ax.set_ylabel('e$_\\mathsf{e}$')
    f.ezlegend(ax, loc=2, title='TTGM-{} Max Values'.format(expt))
    ax.axis(xmin=0,ymin=0)
    f.myax(ax)
    
    fig2, ax2 = p.subplots()
    ax2.plot(-2*X[:,13], X[:,14])
    ax2.set_xlabel('2e$_{\\theta\\mathsf{x}}$')
    ax2.set_ylabel('e$_\\mathsf{xx}$')
    f.eztext(ax2, 'TTGM-{} Max Values'.format(expt), 'ul')
    f.myax(ax2)
    p.show()

def NewOldMax(expt, mode=None):
    '''
    Plots the STK16 max point and the new max point eeq vs rot
    Can plot either Total or incremental method of calculation
    '''
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,5]
    oldmax = n.genfromtxt('../{}/MaxPt.dat'.format(proj), delimiter=',')[:,10]
    oldmax_incr = n.genfromtxt('../{}/IncrementalAnalysis/OldFilteringResults.dat'.format(proj),
                    delimiter=',', usecols=(10))
    maxi, maxj = n.genfromtxt('../{}/max.dat'.format(proj), delimiter=',')[-1,-2:]
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) 
    # [12,13,14,15,16,17] e00, e01, e11, eeqVM, eeqH8, eeqAnis
    D = n.load('../{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(proj))
    locmax = D[-1, :, 15].argmax()

    p.style.use('mysty')
    fig, ax = p.subplots()
    if mode != 'Total':
        ax.plot(dr, oldmax_incr, label='STK16 ({:.0f},{:.0f})'.format(maxi,maxj), zorder=10)
        ax.plot(dr, D[:, locmax, 15], label='New ({:.0f},{:.0f})'.format(*D[-1, locmax, :2]))
        title='TTGM-{} Max Incr. Values'.format(expt)
    else:
        ax.plot(dr, oldmax, label='STK16 ({:.0f},{:.0f})'.format(maxi,maxj), zorder=10)
        # [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
        h = n.genfromtxt('../{}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(proj),
                        delimiter=',', usecols=(13,))
        ax.plot(dr, h, label='New ({:.0f},{:.0f})'.format(*D[-1, locmax, :2]))
        title='TTGM-{} Max Total Stn'.format(expt)

    ax.set_xlabel('$\\Phi$')
    ax.set_ylabel('e$_\\mathsf{e}$')
    ax.axis(xmin=0,ymin=0)
    f.ezlegend(ax, title=title, loc=2)
    f.myax(ax)

    return None

def PlotAllEeqRot(expt=24, ij=None, kelin=False):
    '''
    Plots every maxima's eeq vs rot
    '''
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,5]
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) 
    # [12,13,14,15,16,17] e00, e01, e11, eeqVM, eeqH8, eeqAnis
    d = n.load('../{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(proj))
    locmax = d[-1, :, 15].argmax()  
    p.style.use('mysty')
    fig, ax = p.subplots()
    p.plot(dr, d[:,:,15], alpha=0.2)
    p.plot(dr, d[:,locmax,15], 'k', label='Max ({},{})'.format(*d[-1,locmax,:2].astype(int)))
    p.plot(dr, d[:,:,15].mean(axis=1), 'k--', label='Mean')
    if ij is not None:
        try:
            loc = n.nonzero( (d[-1,:,0]==ij[0]) & (d[-1,:,1]==ij[1]) )[0][0]
            p.plot(dr, d[:,loc,15], 'r', label='({},{})'.format(*ij))
        except IndexError:
            d = n.load('../{}/IncrementalAnalysis/PointsInLastWithStrains.npy'.format(proj))
            loc = n.nonzero( (d[0,:,0]==ij[0]) & (d[0,:,1]==ij[1]) )[0][0]
            ee = []
            for i in range(d.shape[0]):
                ee.append(d[i,loc,15])
            p.plot(dr, ee, 'r', label='({},{}) (no pass)'.format(*ij))
    if kelin and (expt in [1, 4, 5]):
        cols = {4:(0,1), 1:(2,3), 5:(4,5)}
        s = read_excel('../Misc/KelinSim.xlsx', parse_cols=cols[expt]).values
        p.plot(s[:,0], s[:,1], 'b', label='Anal Anis')
    ax.axis(xmin=0,ymin=0)
    ax.set_xlabel('$\\Phi$')
    ax.set_ylabel('e$_\\mathsf{e}$')
    f.ezlegend(ax, loc=2, hl=2, title='TTGM-{}'.format(expt))
    f.myax(ax)  

def KelinCompare(expt, savedata=False):
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    key = n.genfromtxt('../ExptSummary.dat', delimiter=',')
    alpha = key[ key[:,0] == expt, 3].ravel()
    dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,5]
    d = n.load('../{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(proj))
    mnmx = d[...,17].min(axis=1), d[...,17].max(axis=1)
    cols = {4:(0,1), 1:(2,3), 5:(4,5)}
    s = read_excel('../Misc/KelinSim.xlsx', parse_cols=cols[expt]).values
    if savedata:
        n.savetxt('KelinCompare_Expt-{}.dat'.format(expt), X=n.c_[dr, d[...,17].mean(axis=1), mnmx[0], mnmx[1]], fmt="%.6f", delimiter=',')
        n.savetxt('KelinCompare_Expt-{}_kel.dat'.format(expt), X=s, fmt="%.6f", delimiter=',')
    p.style.use('mysty')
    fig, ax = p.subplots()
    p.fill_between(dr, mnmx[0], mnmx[1], alpha=0.1, color='C0')
    p.plot(dr, mnmx[1], 'C0', label='Exp')
    p.plot(dr, mnmx[0], 'C0')
    p.plot(dr, d[:,:,17].mean(axis=1), '--', color='C0')
    p.plot(s[:,0], s[:,1], 'C1', lw=2, label='Anal Anis')
    ax.axis(xmin=0,ymin=0)
    ax.set_xlabel('$\\Phi$')
    ax.set_ylabel('e$_\\mathsf{e}$')
    f.ezlegend(ax, loc=2, title='TTGM-{}\n$\\alpha$ = {:g}'.format(expt, alpha[0]))
    f.myax(ax)      
    return fig, ax

def PlotAllOldFilters(expt):
    p.style.use('mysty-sub')        
    labs = ['Avg_AllPass', 'Avg_P2P', 'LastMax_TraceBk', 'Max_EachStgP2P','LastMax_Trace_Nbhd', 'Avg_P2P_PassOnly']
    alpha = [1,1,1,.5,1,1]
    ls = ['-', '--', '-', '-', '--', '-', '--']
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    dmean = n.genfromtxt('../{}/mean.dat'.format(proj))[:,-1]
    dmax = n.genfromtxt('../{}/MaxPt.dat'.format(proj))[:,10]
    X = n.genfromtxt('../{}/IncrementalAnalysis/OldFilteringResults.dat'.format(proj), delimiter=',')
    if expt in ['18',18]:
        dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,4]
        xlab = '$\\delta/\\mathsf{L}$'
    else:
        dr = n.genfromtxt('../{}/disp-rot.dat'.format(proj), delimiter=',')[:,5]
        xlab = '$\\Phi$'    
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
    p.show()
   
def GetMaxPtF(expt, maxmode='new'):
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) 
    # [12,13,14,15,16] e00, e01, e11, eeqVM, eeqH8    
    d = n.load('../{}/IncrementalAnalysis/PointsInLastWithStrains.npy'.format(proj))
    if maxmode != 'old':
        loc = d[-1,:,15].argmax()
    else:
        maxi, maxj = n.genfromtxt('../{}/max.dat'.format(proj), delimiter=',')[-1,-2:]
        maxp = pair(n.c_[maxi,maxj])
        IDs = pair(d[1,:,:2])
        loc = n.nonzero(IDs == maxp)[0][0]
    F = d[:,loc,8:12]
    return F
    
def PlotF(expt, F=None):
    '''
    Plot Components of the def. grad.
    If F isn't given, it gets the expt's old max point F
    '''
    if F is None:
        F = GetMaxPtF(expt)
    p.plot(-F[:,1], F[:,3], marker='.')
    p.axis(xmin=0, ymin=1)
    p.xlabel('-F$_\\mathsf{01}$')
    p.ylabel('F$_\\mathsf{11}$')
    if expt is not None:
        f.eztext(p.gca(), 'TTGM-{}'.format(expt), 'ul')
    f.myax(p.gca())
    return F
    
def PlotStrainPaths(expt, ij=None, stndef=None, eq=False):
    '''
    Plots all new passing points incr. strain paths.
    Also highlights the max and mean.
    if ij is not None, then it will plot a given ij point's path
    '''
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) 
    # [12,13,14,15,16,17] e00, e01, e11, eeqVM, eeqH8, eeqAnis
    d = n.load('../{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(proj))
    maxloc = d[-1,:,15].argmax()
    maxij = d[-1,maxloc,:2].astype(int)
    # [0-5]Mean VM-H8-Anis-de00-01-11, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
    maxp = n.genfromtxt('../{}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(proj), delimiter=',')
    p.style.use('mysty')
    fig, ax = p.subplots()
    if expt in [18, '18']:
        col = 12
    else:
        col = 13
    if ij is not None:
        loc = n.nonzero( (d[-1,:,0]==ij[0]) & (d[-1,:,1]==ij[1]) )[0][0]
    if stndef not in ['Haltom', 'Aramis']:
        if eq == True:
            # Plots hoops strain vs gamma instead
            d[:,:,12:15] = d[:,:,[14,13,12]]
            maxp[:,[5,10]] = maxp[:,[2,7]]
        [p.plot(-2*d[:,i,col], d[:,i,14], alpha=0.3) for i in range(d.shape[1])]
        p.plot(-2*maxp[:,col-3], maxp[:,11], 'k', label='Max ({},{})'.format(*maxij))
        p.plot(-2*maxp[:,col-9], maxp[:,5], 'k--', label='Mean')
        if ij is not None:
            p.plot(-2*d[:,loc,col], d[:,loc, 14], 'r', label='{:.0f},{:.0f}'.format(*ij))
    elif stndef == 'Haltom':
        [p.plot(-n.arctan2(d[:,i,9],d[:,i,11]), 
                 d[:,i,11] - 1, alpha=0.5) for i in range(d.shape[1])]
        p.plot(-n.arctan2(d[:,maxloc,9], d[:,maxloc,11]),
               d[:,maxloc,11] - 1, 'k', label='Max ({},{})'.format(*maxij))
        p.plot(-n.arctan2(d[:,:,9], d[:,:,11]).mean(axis=1),
                d[:,:,11].mean(axis=1) - 1, 'k--', label='Mean')
        if ij is not None:
            p.plot(-n.arctan2(d[:,loc,9],d[:,loc,11]), 
                    d[:,loc,11] - 1, 'r', label='{:.0f},{:.0f}'.format(*ij))
    elif stndef == 'Aramis':
        tempG = n.empty(d.shape[:2])
        tempNEy = n.empty_like(tempG)
        for i in range(d.shape[1]):
            lep, NEx, NExy, NEy = LEp(d[:,i,8], d[:,i,9], d[:,i,10], d[:,i,11], True)
            NEx-=1
            NEy-=1
            G = -n.arctan(NExy/(1+NEx)) - n.arctan(NExy/(1+NEy))
            tempG[:,i] = G
            tempNEy[:,i] = NEy
            p.plot(G,NEy,alpha=0.5)
        p.plot(tempG[:, maxloc], tempNEy[:,maxloc], 'k', label='Max ({},{})'.format(*maxij))
        p.plot(tempG.mean(axis=1), tempNEy.mean(axis=1), 'k--', label='Mean')
        if ij is not None:
            p.plot(tempG[:, loc], tempNEy[:,loc], 'r', label='{:.0f},{:.0f}'.format(*ij))
    if expt not in [18, '18', 31, '31']:
        p.axis(xmin=0, ymin=0)
    ax.set_xlabel('2e$_{\\theta\\mathsf{x}}$')
    ax.set_ylabel('e$_\\mathsf{xx}$')
    f.ezlegend(ax, loc=2, hl=2, title='TTGM-{}'.format(expt))
    f.myax(ax)
    p.show()
    return fig, ax

def FailureStrains(max=True, mean=True, vm=True, h8=False, anis=False):
    ex = n.genfromtxt('../ExptSummary.dat', delimiter=',')
    ex = ex[ (ex[:,1] == 0) & (ex[:,3]!= 0) ][:,0].astype(int)
    failz = n.genfromtxt('./failstns.dat', delimiter='\t')
    p.style.use('mysty')
    fig, ax = p.subplots()
    for k,x in enumerate(ex):
        lines = []
        triax = failz[ failz[:,0] == x, 2 ]
        proj = 'TTGM-{}_FS19SS6'.format(x)
        # [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
        e = n.genfromtxt('../{}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(proj), delimiter=',')[-1]
        if x == [35]:
            pass
            #e = n.genfromtxt('../{}/IncrementalAnalysis/NewFilterResults_4med.dat'.format(proj), delimiter=',')[-1]
        if vm:
            if mean: lines.append(ax.plot(triax, e[0], 'gs', label='Mean/VM')[0])
            if max: lines.append(ax.plot(triax, e[6], 'rs', label='Max/VM')[0])
        if h8:
            if mean: lines.append(ax.plot(triax, e[1], 'go', mfc='none', label='Mean/H8')[0])
            if max: lines.append(ax.plot(triax, e[7], 'ro', mfc='none', label='Max/H8')[0])
        if anis:
            if mean: lines.append(ax.plot(triax, e[2], 'g^', mfc='none', label='Mean/Ani')[0])
            if max:lines.append(ax.plot(triax, e[8], 'r^', mfc='none', label='Max/Ani')[0])        
    
    leg = ax.legend(lines, [i.get_label() for i in lines], loc=1)
    [l.set_color(leg.get_lines()[k].get_mec()) for k,l in enumerate(leg.get_texts())]
    ax.axis(xmin=0,ymin=0,ymax=1.65)
    ax.set_xlabel('$\\sigma_{\\mathsf{m}}/\\sigma_{\\mathsf{e}}$')
    ax.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
    f.eztext(ax, 'Incremental\nAl-6061-T6', 'bl')
    f.myax(ax)
    return fig, ax

def AllMaxesTriax(constit='vm', savedata=False):
    '''
    Plots the vm eeq at failure for every passing column max
    '''
    ex = n.genfromtxt('../ExptSummary.dat', delimiter=',')
    ex = ex[ (ex[:,1] == 0) & (ex[:,3]!= 0) ][:,0].astype(int)
    failz = n.genfromtxt('./failstns.dat', delimiter='\t')
    col = {'vm':15, 'h8':16, 'anis':17}
    p.style.use('mysty')
    fig, ax = p.subplots()
    for k,x in enumerate(ex):
        lines = []
        triax = failz[ failz[:,0] == x, 2 ]
        proj = 'TTGM-{}_FS19SS6'.format(x)    
        d = n.load('../{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(proj))[-1,:,col[constit]]
        if x == 10:
            d = d[ d!= d.min() ]
        p.plot(n.ones_like(d)*triax, d, '.', linestyle='none', color='C0')
        p.plot(triax, d.mean(), 'o', color='C1')
    ax.axis(xmin=0)
    ax.set_xlabel('$\\sigma_{\\mathsf{m}}/\\sigma_{\\mathsf{e}}$')
    ax.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
    f.eztext(ax, 'Incremental\nAl-6061-T6', 'bl')
    f.myax(ax)
    p.show()
    return fig, ax        

def SetStrainpaths(Mean=True,Max=False):
    '''
    Plot axial vs shear strain (mean or max) for all expts)
    '''
    ex = n.genfromtxt('../ExptSummary.dat', delimiter=',')
    ex = ex[ ex[:,3].argsort() ][::-1]
    ex = ex[ (ex[:,3]!= 0) ]
    alp = ex[:,3]
    ex = ex[:,0].astype(int)
    p.style.use('mysty')
    fig, ax = p.subplots()
    for k,(x,a) in enumerate(zip(ex,alp)):
        proj = 'TTGM-{}_FS19SS6'.format(x)    
        LL = n.genfromtxt('../{}/prof_stages.dat'.format(proj), delimiter=',', dtype=int)[3]
        # [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
        D = n.genfromtxt('../{}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(proj),
                        delimiter=',')
        D[:,[4,10]]*=-1
        if n.isnan(a):
            a = '$\\infty$'
        line, = p.plot(D[:,4], D[:,5], label='{}'.format(a))
        LLmark, = p.plot(D[LL,4], D[LL,5], 'rs')
        if Max:
            maxline, = p.plot(D[:,10], D[:,11], color=line.get_color(), alpha=0.3)
            maxmark, = p.plot(D[LL,10], D[LL,11], 'rs', alpha=0.3)
        if not Mean:
            maxline.set_alpha(1)
            maxmark.set_alpha(1)
            maxline.set_label(line.get_label())
            line.remove()
            LLmark.remove()
    ax.axis(xmin=-.02, ymin=0)
    ax.set_xlabel('e$_{\\theta\\mathsf{x}}$')
    ax.set_ylabel('e$_\\mathsf{xx}$')
    f.ezlegend(ax, title='Mean'*Mean+'\n'*(Max&Mean)+'Max'*Max, loc=0)
    f.myax(ax)
    p.show()

    
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

def increm_rot(A,B):
    '''
    Requires A and B be (nx4), with cols corresponding to F00, F01, F10, F11
    This is the 01 component of the incremental spin tensor
    '''
    dr = ((-(A[:,0] - B[:,0])*(A[:,1] + B[:,1]) + (A[:,0] + B[:,0])*(A[:,1] - B[:,1]) -
           (A[:,2] - B[:,2])*(A[:,3] + B[:,3]) + (A[:,2] + B[:,2])*(A[:,3] - B[:,3]))/
           ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
         )
    return -dr  

def deeq(E,sig):
    '''
    E must be  nx3, columns corresponding to de00, de01, de11
    sig must be a list or tuple of (sig00, sig01, sig11, sigeq)
    '''
    return (E[:,0]*sig[0]+E[:,2]*sig[2]-2*E[:,1]*sig[1])/sig[3]

def pair(D):
    '''
    Cantor pairing function from wikipedia
    D must be (nx2)
    '''
    if (D.ndim != 2) and (D.shape[1] != 2):
        raise ValueError('Array must be nx2')
    else:
        return (D[:,0] + D[:,1]) * (D[:,0] + D[:,1] + 1)/2 + D[:,1]    
    
def MaxInPassing(expt, maxij=None):
    from scipy.io import loadmat
    '''
    Tells you which stages of old filtering the STK16 max point passed 
    '''
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    maxes = n.genfromtxt('../{}/max.dat'.format(proj), delimiter=',')
    num = maxes.shape[0]
    if maxij is None:
        maxi, maxj = maxes[-1,-2:]
    else:
        # Then I've passed in a tuple of points
        maxi, maxj = maxij
    maxp = pair(n.c_[maxi,maxj])
    pp = loadmat('../{0}/{0}_PassingPts.mat'.format(proj))
    ins = []
    for i in range(num):
        IDp = pair(pp['stage_{}'.format(i)][:,:2])
        if maxp in IDp:
            ins.append(i)
    return n.array(ins)    
    
def PlotOldStrainDef(expt, stndef='Haltom'):
    '''
    Obsolete.  See PlotStrainPaths
    '''
    from FtfEig import LEp
    proj = 'TTGM-{}_FS19SS6'.format(expt)
    # [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
    # [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) 
    # [12,13,14,15,16] e00, e01, e11, eeqVM, eeqH8
    d = n.load('../{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(proj))
    # [0-4]Mean VM-H8-de00-01-00, [5-9]Max VM-H8-de00-01-00, [10-11]Mean, max Classic LEp
    maxp = n.genfromtxt('../{}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(proj), delimiter=',')
    p.style.use('mysty')
    fig, ax = p.subplots()
    if expt in [18, '18']:
        col = 12
    else:
        col = 13
    
    if stndef == 'Haltom':
        # Haltom definitions
        [p.plot(-n.arctan2(d[:,i,9],d[:,i,11]), 
                    d[:,i,11] - 1, alpha=0.5) for i in range(d.shape[1])]
    else:
        # Aramis definition
        for i in range(d.shape[1]):
            lep, NEx, NExy, NEy = LEp(d[:,i,8], d[:,i,9], d[:,i,10], d[:,i,11], True)
            NEx-=1
            NEy-=1
            G = n.arctan(NExy/(1+NEx)) + n.arctan(NExy/(1+NEy))
            p.plot(-G,NEy,alpha=0.5)
    
    p.axis(xmin=0, ymin=0)
    p.xlabel('$\\gamma$')
    p.ylabel('$\\epsilon_\\mathsf{x}$')
    f.myax(ax)
    return fig, ax
    
p.close('all')
