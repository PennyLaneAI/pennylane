import numpy as np
import math


def coeffsNIproc(o: int, s: int, identifier = None, aba = 1):
    r"""
    Returns the coefficients of integrators designed for
    near integrable systems.

    Parameters
    -----------
    o: int
        order of the integrator
    s: int
        Number of stages
    identifier:
        Unique identifier given to the method.
        Necessary if there are multiple methods with same s and o

    Returns
    -----------
    lists

    Notes:
    -----------
    Check https://arxiv.org/abs/2401.01722, table 8.5 for the identifiers
    and recommended methods.
    """
    if (o[0]==2 and o[1]==2):
        s==1
        identifier == 'Strang'
        if (aba==1):
            a=[1/2 , 1/2]
            b=[1 , 0]
    elif (o[0]==6 and o[1]==4):
        # ABA method
        s=2
        a1=(1+math.sqrt(1+2/math.sqrt(3)))/2
        a2=1-a1
        b1=(1-math.sqrt(1+2/math.sqrt(3)))/2
        b2=1-b1
        a=[a2, a1]
        b=[b2, b1]

        z1= 0 
        y1=-1/2
        z2=-0.4842296798457861 
        y2=-0.071444491827245766
        z3=0.3104400973059574 
        y3=0.5119423407689261
        z4=0.9112603236884162 
        y4=-0.2256471548948410
        ap=[z1, z2, z3, z4]
        bp=[y1, y2, y3, y4]
    elif  (o[0]==7 and o[1]==6):
        if aba: # ABA method
            s = 3
            a1=0.5600879810924619 
            b1=1.5171479707207228
            a2=1/2-a1        
            b2 = 1-2*(b1) 
            a=[a1, a2, a2, a1]
            b=[b1, b2, b1,  0]
            
            z1=-0.3346222298730800 
            y1=-1.621810118086801
            z2= 1.097567990732164 
            y2= 0.0061709468110142
            z3=-1.038088746096783 
            y3= 0.8348493592472594
            z4= 0.6234776317921379 
            y4=-0.0511253369989315
            z5=-1.102753206303191 
            y5= 0.5633782670698199
            z6=-0.0141183222088869 
            y6=-1/2    
            ap=[z1, z2, z3, z4, z5, z6]
            bp=[y1, y2, y3, y4, y5, y6]
        else: # BAB method
            s = 4
            a1=-0.6659279171311354 
            b1= 0.0962499147414666
            b2=-0.0649951074268679   
            a2=1/2-a1        
            b3 = 1-2*(b1+b2) 
            a=[0,  a1, a2, a2, a1]
            b=[b1, b2, b3, b2, b1]

            z1=-0.5682049251492933
            y1= 0.2005780724079215
            z2= 0.2817876004745961 
            y2=-0.3923456667727871
            z3= 0.7168960305523042 
            y3=-0.9517071967056039
            z4= 0.4332386614652446 
            y4=-0.0443156930081850
            z5=-0.3552157340165512 
            y5= 0.7361124293734198
            z6=-0.5825683076056897 
            y6= 0.3776113804258454    
            ap=[z1, z2, z3, z4, z5, z6]
            bp=[y1, y2, y3, y4, y5, y6]

    return a, b, ap, bp, s