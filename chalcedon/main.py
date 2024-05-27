import numpy as np
import scipy
import skimage

import tdpy
import aspendos


def retr_radieins_inft( \
                       # velocity dispersion [km/s]
                       dispvelo, \
                      ):
    '''
    Calculate the Einstein radius for a source position at infinity
    '''
    
    radieins = 4 * np.pi * (dispvelo / 3e5)**2 * (180. / np.pi * 3600.) # [arcsec]
    
    return radieins


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


def retr_deflcutf(xposgrid, yposgrid, dictchalinpt, boolasym=False):
    '''
    Return deflection due to a subhalo with a cutoff radius
    '''
    
    distxpos = xposgrid[:, None] - dictchalinpt['xpossubh'][None, :]
    distypos = yposgrid[:, None] - dictchalinpt['ypossubh'][None, :]
    
    distangl = np.sqrt(distxpos**2 + distypos**2)
    
    fracanglasca = distangl / dictchalinpt['ascasubh']
    
    deflcutf = defs / fracanglasca
    
    # second term in the NFW deflection profile
    fact = np.ones_like(fracanglasca)
    indxlowr = np.where(fracanglasca < 1.)[0]
    indxuppr = np.where(fracanglasca > 1.)[0]
    fact[indxlowr] = np.arccosh(1. / fracanglasca[indxlowr]) / np.sqrt(1. - fracanglasca[indxlowr]**2)
    fact[indxuppr] = np.arccos(1. / fracanglasca[indxuppr]) / np.sqrt(fracanglasca[indxuppr]**2 - 1.)
    
    if boolasym:
        deflcutf *= np.log(fracanglasca / 2.) + fact
    else:
        fracacutasca = acut / asca
        factcutf = fracacutasca**2 / (fracacutasca**2 + 1)**2 * ((fracacutasca**2 + 1. + 2. * (fracanglasca**2 - 1.)) * fact + \
                np.pi * fracacutasca + (fracacutasca**2 - 1.) * np.log(fracacutasca) + \
                np.sqrt(fracanglasca**2 + fracacutasca**2) * (-np.pi + (fracacutasca**2 - 1.) / fracacutasca * \
                np.log(fracanglasca / (np.sqrt(fracanglasca**2 + fracacutasca**2) + fracacutasca))))
        deflcutf *= factcutf
       
    return deflcutf


def retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt):
    '''
    Return deflection due to a main halo without a cutoff radius and subhalos with cutoff
    '''
    
    dictchaloutp = dict()

    # check inputs
    if dictchalinpt['ellphalo'] is not None and (dictchalinpt['ellphalo'] < 0. or dictchalinpt['ellphalo'] > 1.):
        raise Exception('')

    # translate the grid
    xposgridtran = xposgrid[indxpixlelem] - dictchalinpt['xposhalo']
    yposgridtran = yposgrid[indxpixlelem] - dictchalinpt['yposhalo']
    
    anglgrid = np.sqrt(xposgridtran**2 + yposgridtran**2)
    
    # rotate the grid
    xposgridrttr = np.cos(anglgrid) * xposgridtran - np.sin(anglgrid) * yposgridtran
    yposgridrttr = np.sin(anglgrid) * xposgridtran + np.cos(anglgrid) * yposgridtran
    
    axisrati = 1. - dictchalinpt['ellphalo']
    facteccc = np.sqrt(1. - axisrati**2)
    factrcor = np.sqrt(axisrati**2 * xposgridrttr**2 + yposgridrttr**2)
    
    deflxposrttr = dictchalinpt['beinhalo'] * axisrati / facteccc *  np.arctan(facteccc * xposgridrttr / factrcor)
    deflyposrttr = dictchalinpt['beinhalo'] * axisrati / facteccc * np.arctanh(facteccc * yposgridrttr / factrcor)
    
    # rotate back vector to original basis
    dictchaloutp['deflxposhalo'] = np.cos(anglgrid) * deflxposrttr + np.sin(anglgrid) * deflyposrttr
    dictchaloutp['deflyposhalo'] = -np.sin(anglgrid) * deflxposrttr + np.cos(anglgrid) * deflyposrttr
   
    dictchaloutp['deflhalo'] = np.vstack((dictchaloutp['deflxposhalo'], dictchaloutp['deflyposhalo'])).T
    
    if 'xpossubh' in dictchalinpt:
        defl = retr_deflcutf(xposgrid, yposgrid, dictchalinpt)
        dictchaloutp['deflxpossubh'] = xposgridtran / anglgrid * defl
        dictchaloutp['deflypossubh'] = yposgridtran / anglgrid * defl
        
        dictchaloutp['deflhalo'] += dictchaloutp['deflsubh']

    return dictchaloutp


def retr_caustics(xposgrid, yposgrid, indxpixlelem, dictchalinpt):
   
    from skimage import measure
    
    deflmain = retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt)
    deflsubh = retr_
    magn = retr_magn()
    
    cont = measure.find_contours(magn, 0.8)
    print('cont')
    print(cont)
