from numpy import sqrt, log

def Eig2x2(a,b,c,d):
    '''
    Mx must be [[a,b],[c,d]]
    '''
    return (a + d - sqrt(a**2 + 4*b*c - 2*a*d + d**2))/2, (a + d + sqrt(a**2 + 4*b*c - 2*a*d + d**2))/2

def LEp(F00, F01, F10, F11, rtnU=False):
    U00 = (sqrt(F00**2 + F01**2 + F10**2 + F11**2 - sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
           ((F01 - F10)**2 + (F00 + F11)**2)))*(-F00**2 + F01**2 - F10**2 + F11**2 + 
        sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))) + 
      (F00**2 - F01**2 + F10**2 - F11**2 + sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2)))*sqrt(F00**2 + F01**2 + F10**2 + F11**2 + 
         sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))))/(
     (2*sqrt(2)*sqrt(-4*(F01*F10 - F00*F11)**2 + (F00**2 + F01**2 + F10**2 + F11**2)**2)))


    U01 = (sqrt(2)*(F00*F01 + F10*F11))/(
     (sqrt(F00**2 + F01**2 + F10**2 + F11**2 - sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2))) + 
      sqrt(F00**2 + F01**2 + F10**2 + F11**2 + sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2)))))


    U11 = (sqrt(F00**2 + F01**2 + F10**2 + F11**2 - sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
           ((F01 - F10)**2 + (F00 + F11)**2)))*(F00**2 - F01**2 + F10**2 - F11**2 + 
        sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))) + 
      (-F00**2 + F01**2 - F10**2 + F11**2 + sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2)))*sqrt(F00**2 + F01**2 + F10**2 + F11**2 + 
         sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))))/(
     (2*sqrt(2)*sqrt(-4*(F01*F10 - F00*F11)**2 + (F00**2 + F01**2 + F10**2 + F11**2)**2)))
     
    eigU = Eig2x2(U00, U01, U01, U11)
    LE0, LE1 = [log(u) for u in eigU]
    
    if rtnU != True:
        return ( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5
    else:
        return ( ( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5,
                U00, U01, U11
                )

