import numpy as np
import scipy

import tdpy
import aspendos


def retr_radieins_inft( \
                       # velocity dispersion [km/s]
                       dispvelo, \

                      ):
    '''
    Calculate the Einstein radius for a source position at infinity
    '''
    """
            :param deflector_dict: deflector properties
            :param v_sigma: velocity dispersion in km/s
            :return: Einstein radius in arc-seconds
            """
    if v_sigma is None:
        if deflector_dict is None:
            raise ValueError("Either deflector_dict or v_sigma must be provided")
        else:
            v_sigma = deflector_dict['vel_disp']

    theta_E_infinity = 4 * np.pi * (dispvelo / 3e5)**2 * (180. / np.pi * 3600.)
    return theta_E_infinity


def retr_dflxslensing(time, epocslen, amplslen, duratrantotl):
    '''
    Return the self-lensing signature
    '''    
    
    timediff = time - epocslen
    
    dflxslensing = 1e-3 * amplslen * np.heaviside(duratrantotl / 48. + timediff, 0.5) * np.heaviside(duratrantotl / 48. - timediff, 0.5)
    
    return dflxslensing


def retr_angleinscosm(masslens, distlenssour, distlens, distsour):
    '''
    Return Einstein radius for a cosmological source and lens.
    '''
    
    angleins = np.sqrt(masslens / 10**(11.09) * distlenssour / distlens / distsour)
    
    return angleins


def retr_radieinstinft(dispvelo):
    '''
    Return Einstein radius for a source at infinity
    '''
    
    radieinstinft = 4 * np.pi * (dispvelo / 3e5)**2 / np.pi * 180. * 3600. # [arcsec]
    
    return radieinstinft


def retr_amplslen( \
                  # orbital period [days]
                  peri, \
                  # radistar: radius of the star [Solar radius]
                  radistar, \
                  # mass of the companion [Solar mass]
                  masscomp, \
                  # mass of the star [Solar mass]
                  massstar, \
                 ):
    '''
    Calculate the self-lensing amplitude.
    '''
    
    amplslen = 7.15e-5 * radistar**(-2.) * peri**(2. / 3.) * masscomp * (masscomp + massstar)**(1. / 3.) * 1e3 # [ppt]

    return amplslen


def retr_radieinssbin(masslens, distlenssour):
    '''
    Return Einstein radius for a stellar lens and source in proximity.
    '''
    
    radieins = 0.04273 * np.sqrt(masslens * distlenssour) # [R_S]
    
    return radieins


def retr_factmcutfromdefs(adissour, adislens, adislenssour, asca, acut):
    
    mdencrit = retr_mdencrit(adissour, adislens, adislenssour)
    
    fracacutasca = acut / asca
    
    factmcutfromdefs = np.pi * adislens**2 * mdencrit * asca * aspendos.retr_mcutfrommscl(fracacutasca)

    return factmcutfromdefs


def retr_mdencrit(adissour, adislens, adislenssour):
    '''
    Calculate the critical mass density at a given angular diameter distance to the source, to the lens, and between the lens and the source.
    '''
    
    dictfact = tdpy.retr_factconv()
    
    mdencrit = dictfact['factnewtlght'] / 4. / np.pi * adissour / adislenssour / adislens
        
    return mdencrit


def retr_ratimassbeinsqrd(adissour, adislens, adislenssour):
    '''
    Calculate the ratio between the mass of the lens and square of its Einstein radius
    '''
    
    # calculate the critical mass density
    mdencrit = retr_mdencrit(adissour, adislens, adislenssour)
    
    # ratio between the mass of the lens and square of its Einstein radius
    ratimassbeinsqrd = np.pi * adislens**2 * mdencrit

    return ratimassbeinsqrd


def retr_deflextr(xposgrid, yposgrid, sher, sang):
    '''
    Return deflection field due to large-scale structure
    '''    
    
    factcosi = sher * np.cos(2. * sang)
    factsine = sher * np.cos(2. * sang)
    deflxpos = factcosi * xposgrid + factsine * yposgrid
    deflypos = factsine * xposgrid - factcosi * yposgrid
    
    deflextr = np.vstack((deflxpos, deflypos)).T

    return deflextr


def retr_deflcutf(angl, defs, asca, acut, asym=False):
    '''
    Return deflection due to a subhalo with a cutoff radius
    '''

    fracanglasca = angl / asca
    
    deflcutf = defs / fracanglasca
    
    # second term in the NFW deflection profile
    fact = np.ones_like(fracanglasca)
    indxlowr = np.where(fracanglasca < 1.)[0]
    indxuppr = np.where(fracanglasca > 1.)[0]
    fact[indxlowr] = np.arccosh(1. / fracanglasca[indxlowr]) / np.sqrt(1. - fracanglasca[indxlowr]**2)
    fact[indxuppr] = np.arccos(1. / fracanglasca[indxuppr]) / np.sqrt(fracanglasca[indxuppr]**2 - 1.)
    
    if asym:
        deflcutf *= np.log(fracanglasca / 2.) + fact
    else:
        fracacutasca = acut / asca
        factcutf = fracacutasca**2 / (fracacutasca**2 + 1)**2 * ((fracacutasca**2 + 1. + 2. * (fracanglasca**2 - 1.)) * fact + \
                np.pi * fracacutasca + (fracacutasca**2 - 1.) * np.log(fracacutasca) + \
                np.sqrt(fracanglasca**2 + fracacutasca**2) * (-np.pi + (fracacutasca**2 - 1.) / fracacutasca * \
                np.log(fracanglasca / (np.sqrt(fracanglasca**2 + fracacutasca**2) + fracacutasca))))
        deflcutf *= factcutf
       
    return deflcutf


def retr_defl(xposgrid, yposgrid, indxpixlelem, xpos, ypos, angllens, ellp=None, angl=None, rcor=None, asca=None, acut=None):
    '''
    Return deflection due to a halo without a cutoff radius
    '''

    # translate the grid
    xpostran = xposgrid[indxpixlelem] - xpos
    ypostran = yposgrid[indxpixlelem] - ypos
    
    if ellp is not None and (ellp < 0. or ellp > 1.):
        raise Exception('')

    if acut is not None:
        defs = angllens
        angl = np.sqrt(xpostran**2 + ypostran**2)
        defl = retr_deflcutf(angl, defs, asca, acut)
        deflxpos = xpostran / angl * defl
        deflypos = ypostran / angl * defl

    else:
        bein = angllens

        # rotate the grid
        xposrttr = np.cos(angl) * xpostran - np.sin(angl) * ypostran
        yposrttr = np.sin(angl) * xpostran + np.cos(angl) * ypostran
        
        axisrati = 1. - ellp
        facteccc = np.sqrt(1. - axisrati**2)
        factrcor = np.sqrt(axisrati**2 * xposrttr**2 + yposrttr**2)
        
        deflxposrttr = bein * axisrati / facteccc *  np.arctan(facteccc * xposrttr / factrcor)
        deflyposrttr = bein * axisrati / facteccc * np.arctanh(facteccc * yposrttr / factrcor)
        
        # totate back vector to original basis
        deflxpos = np.cos(angl) * deflxposrttr + np.sin(angl) * deflyposrttr
        deflypos = -np.sin(angl) * deflxposrttr + np.cos(angl) * deflyposrttr
   
    defl = np.vstack((deflxpos, deflypos)).T
    
    return defl


def retr_caustics(dictchalinpt):

    pass

