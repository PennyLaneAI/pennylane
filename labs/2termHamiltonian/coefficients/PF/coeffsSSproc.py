import numpy as np

def coeffsSSproc(o: int, s: int, identifier = None):
    r"""
    Returns the coefficients of integrators made up of symmetric compositions
    of symmetric steps with processing.

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
    k, p, np
        k: list
            Coefficients of the kernel.
        p: list
            Coefficients of the processor.
        np: int
            Number of stages in the processor.

    Notes:
    -----------
    Check https://arxiv.org/abs/2401.01722, table 8.1 for the identifiers
    and recommended methods.
    """
    if o==6 and s==7 and identifier == 'BLA01':
        a1= 0.513910778424374 
        a2= 0.364193022833858
        a3= -0.867423280969274
        k =[a1, a2,a3, 1-2*(a1+a2+a3), a3, a2, a1]

        n_p=10

        ga5= 0.375012038697862
        ga4= 0.384998538774070
        ga3=-0.074332422810238 
        ga2=-0.461165940466494
        ga1=-(ga2+ga3+ga4+ga5)
        ga6=-ga1
        ga7=-ga2 
        ga8=-ga3 
        ga9=-ga4 
        ga10=-ga5
        p=-1*np.array([ga1,ga2,ga3,ga4,ga5,ga6,ga7,ga8,ga9,ga10])
    
    elif o==6 and s==11 and identifier == 'BCM06':

        a1= 0.1705768865009222157
        a5= -0.423366140892658048
        k=[a1, a1, a1, a1, a5, 1-2*(4*a1+a5), a5, a1, a1, a1, a1]

        n_p=12

        ga6=-0.1
        ga5= 0.24687306977659
        ga4= 0.09086982276241
        ga3=0.23651387483203 
        ga2=-0.20621953139126
        ga1=-(ga6+ga2+ga3+ga4+ga5)
        ga7=-ga1
        ga8=-ga2
        ga9=-ga3
        ga10=-ga4
        ga11=-ga5
        ga12=-ga6
        p=-1*np.array([ga1,ga2,ga3,ga4,ga5,ga6,ga7,ga8,ga9,ga10,ga11,ga12])

    elif o==6 and s==13 and identifier == 'BCM06':
        a1= 0.125696288720106 
        a5= 0.148070660114965 
        a6=-0.350856370823828
        k=[a1, a1,a1,a1,a5,a6, 1-2*(4*a1+a5+a6), a6, a5, a1, a1, a1, a1]

        n_p=12
        
        ga1= 0.1 
        ga2= 0.225080298761176 
        ga3= 0.191244694511161
        ga4=-0.212763792194890 
        ga5=-0.09660157306582295 
        ga6=-(ga1+ga2+ga3+ga4+ga5)
        ga7=-ga1 
        ga8=-ga2 
        ga9=-ga3
        ga10=-ga4 
        ga11=-ga5 
        ga12=-ga6
        p=-1*np.array([ga1,ga2,ga3,ga4,ga5,ga6,ga7,ga8,ga9,ga10,ga11,ga12])

    elif o==8 and s==13 and identifier == 'BCM06':
        a1= 0.168 
        a3= 0.585550530805562 
        a4= -0.460090457516872 
        a5= 0.172863148729731 
        a6=0.179664539695039
        k=[a1, a1,a3,a4,a5,a6, 1-2*(2*a1+a3+a4+a5+a6), a6, a5, a4, a3, a1, a1]

        n_p=20

        ga10= -0.543415765371656 
        ga9= 0.598212975943381 
        ga8= 0.236885952363384
        ga7=-0.511744926116413 
        ga6= 0.162324207599241
        ga5= 0.588351189003849
        ga4= 0.333987768164597
        ga3= -0.337188967354338
        ga2= -0.008488123494574411
        ga1=-(ga10+ga2+ga3+ga4+ga5+ga6+ga7+ga8+ga9)
        ga11=-ga1 
        ga12=-ga2 
        ga13=-ga3 
        ga14=-ga4 
        ga15=-ga5 
        ga16=-ga6
        ga17=-ga7
        ga18=-ga8
        ga19=-ga9
        ga20=-ga10
        p=-1*np.array([ga1,ga2,ga3,ga4,ga5,ga6,ga7,ga8,ga9,ga10,ga11,
            ga12,ga13,ga14,ga15,ga16,ga17,ga18,ga19,ga20])
        
    elif o==10 and s==23 and identifier == 'BCM06':
        a1= 0.121657748919383 
        a6= -0.511318780154828 
        a7= -0.172858614884985 
        a8=0.123016258833066
        a9=0.441503951671565
        a10=-0.327071324165477
        a11=0.070952700957766
        k=[a1, a1,a1,a1,a1,a6,a7,a8,a9,a10,a11,
            1-2*(5*a1+a6+a7+a8+a9+a10+a11),
            a11,a10,a9,a8,a7,a6,a1,a1,a1,a1,a1]

        n_p=24

        ga12= 0.009116042043427756 
        ga11= 0.5334030283695922 
        ga10= 0.3430345669677392 
        ga9= -0.3594148033156072 
        ga8= 0.1548256472553489
        ga7=-0.1899795533199732 
        ga6= -0.5382945821834320
        ga5= -0.3045590922565247
        ga4= -0.4637104712987078
        ga3= 0.01344750613191108
        ga2= 0.4727142080578221
        ga1=-(ga2+ga3+ga4+ga5+ga6+ga7+ga8+ga9+ga10+ga11+ga12)
        ga13=-ga1 
        ga14=-ga2 
        ga15=-ga3 
        ga16=-ga4 
        ga17=-ga5 
        ga18=-ga6
        ga19=-ga7
        ga20=-ga8
        ga21=-ga9
        ga22=-ga10
        ga23=-ga11
        ga24=-ga12
        p=-1*np.array([ga1,ga2,ga3,ga4,ga5,ga6,ga7,ga8,ga9,ga10,ga11,ga12,
            ga13,ga14,ga15,ga16,ga17,ga18,ga19,ga20,ga21,ga22,ga23,ga24])

    return k, p, n_p
