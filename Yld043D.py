import numpy as n
from numpy import hstack, vstack, dstack
import matplotlib.pyplot as p

'''
'''

def CoefArray(c12,c13,c21,c23,c31,c32,c44,c55,c66):
    C = n.zeros((6,6))
    C[0,1], C[0,2] = -c12, -c13
    C[1,0], C[1,2] = -c21, -c23
    C[2,0], C[2,1] = -c31, -c32
    C[3,3], C[4,4], C[5,5] = c44, c55, c66
    return C

def PHI(Sq, Sx, Sqx, coefs=None, a=8):
    
    if coefs is None:
        '''
        #Older constants c. May 2018
        (cp12,cp13,cp21,cp23,
         cp31,cp32,cp44,cp55,cp66) = (0.6553,0.34303,0.95399,1.2779,
                                       0.73534,1.0449,1.2356,1,1)
        (cpp12,cpp13,cpp21,cpp23,
         cpp31,cpp32,cpp44,cpp55,cpp66) = (1.2531,1.4264,0.73178,0.39816,
                                            0.7372,0.76182,0.82781,1,1)
        
        '''
        # Coeffs as of 08/06/18
        (cp12,cp13,cp21,cp23,
         cp31,cp32,cp44,cp55,cp66) = (0.6787,0.98478,1.1496,1.0275,
                                      0.94143, 1.1621,1.3672,1,1)
        (cpp12,cpp13,cpp21,cpp23,
         cpp31,cpp32,cpp44,cpp55,cpp66) = (1.0563,0.96185,0.68305,0.71287,
                                       1.0933,0.84744,0.69489,1,1)
        
    else:
        (cp12,cp13,cp21,cp23,
             cp31,cp32,cp44,cp55,cp66) = [1 for i in range(9)]
        (cpp12,cpp13,cpp21,cpp23,
            cpp31,cpp32,cpp44,cpp55,cpp66) = [1 for i in range(9)]

    Cp = CoefArray(cp12,cp13,cp21,cp23,cp31,cp32,cp44,cp55,cp66)
    Cpp = CoefArray(cpp12,cpp13,cpp21,cpp23,cpp31,cpp32,cpp44,cpp55,cpp66)

    T = n.zeros_like(Cp)
    T[0,0], T[0,1], T[0,2] = 2, -1, -1
    T[1,0], T[1,1], T[1,2] = -1, 2, -1
    T[2,0], T[2,1], T[2,2] = -1, -1, 2
    T[3,3], T[4,4], T[5,5] = 3, 3, 3
    T/=3

    # Vectorized!
    zero = n.zeros_like(Sq)
    # (6x6)*(6x6)*(n*6)
    Sp = n.einsum('ij,jk,...k', Cp, T, n.c_[zero,Sq,Sx,zero,zero,Sqx])
    Spp = n.einsum('ij,jk,...k', Cpp, T, n.c_[zero,Sq,Sx,zero,zero,Sqx])
    # Take the Sp vector and turn it into a stack of 3x3 pizza boxes
    Sp =  hstack((dstack( (Sp[:,[[0]]],Sp[:,[[3]]], Sp[:,[[4]]]) ),
                  dstack( (Sp[:,[[3]]],Sp[:,[[1]]], Sp[:,[[5]]]) ),
                  dstack( (Sp[:,[[4]]],Sp[:,[[5]]], Sp[:,[[2]]]) )
                  ))
    Spp =  hstack((dstack( (Spp[:,[[0]]],Spp[:,[[3]]], Spp[:,[[4]]]) ),
                  dstack( (Spp[:,[[3]]],Spp[:,[[1]]], Spp[:,[[5]]]) ),
                  dstack( (Spp[:,[[4]]],Spp[:,[[5]]], Spp[:,[[2]]]) )
                  ))
    Sp = n.linalg.eigvalsh(Sp)
    Spp = n.linalg.eigvalsh(Spp)

    PHI = ( (Sp[:,0]-Spp[:,0])**a + 
      (Sp[:,0]-Spp[:,1])**a + 
      (Sp[:,0]-Spp[:,2])**a + 
      (Sp[:,1]-Spp[:,0])**a + 
      (Sp[:,1]-Spp[:,1])**a + 
      (Sp[:,1]-Spp[:,2])**a + 
      (Sp[:,2]-Spp[:,0])**a + 
      (Sp[:,2]-Spp[:,1])**a + 
      (Sp[:,2]-Spp[:,2])**a
    )   
    
    return (0.25*PHI)**.125
    
def CalContour(ax=None,N=100,coefs=None,a=8,close=False):
    '''
    Just return the locus as x,y coords.  Don't plot
    '''
    X,Y = n.mgrid[0:1.2:N*1j, 0:1.2:N*1j]
    Z = PHI(X.ravel(),Y.ravel(),n.zeros(X.size), coefs, a=a)
    if ax is None:
        fig, ax = p.subplots()
    else:
        p.sca(ax)
    c = p.tricontour(Y.ravel(),X.ravel(),Z.ravel(),levels=[1])
    if close == True:
        p.close()
    return c.allsegs[0][0]    

def YldLocusPlot():
    p.style.use('mysty')
    fig = p.figure(figsize=(8,8))
    ax = fig.add_axes([.125,.125,.75,.75])
    anx, any = CalContour(close=True).T
    hx,hy = CalContour(coefs=1,a=8,close=True).T
    vx,vy = CalContour(coefs=1,a=2,close=True).T
    ax.plot(vx,vy,label='VM')
    ax.plot(hx,hy,label='H8')
    ax.plot(anx,any,label='Anis')
    ax.plot([0,1.2],[0,1.2], 'k--', alpha=0.2)
    ax.set_xlabel('$\\tau_\\mathsf{x}/\\tau_o$')
    ax.set_ylabel('yo')
    ax.set_ylabel('$\\frac{\\tau_\\theta}{\\tau_o}$')
    import figfun as f
    f.ezlegend(ax,loc=2)
    ax.set_title("Kelin's Calibration")
    f.myax(ax, autoscale=.9)
        
