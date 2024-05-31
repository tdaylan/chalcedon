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


def retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt):
    '''
    Return deflection due to a main halo without a cutoff radius and subhalos with cutoff radii
    '''
    
    dictchaloutp = dict()

    # check inputs
    if dictchalinpt['ellphalo'] is not None and (dictchalinpt['ellphalo'] < 0. or dictchalinpt['ellphalo'] > 1.):
        raise Exception('')
    
    numbiter = 1
    if 'xpossubh' in dictchalinpt:
        numbsubh = dictchalinpt['xpossubh'].size
        numbiter += numbsubh
    
    print('temp: find out boolasym')
    dictchalinpt['boolasym'] = True
    
    indxiter = np.arange(numbiter)
    for u in indxiter:
        
        if u == 0:
            strgcomp = 'halo'
        else:
            k = u - 1
            strgcomp = 'subh%08d' % k
        
        # dictionary of parameter names for the component
        dictstrg = dict()
        for name in ['xpos', 'ypos', 'ellp', 'bein', 'deflxpos', 'deflypos']:
            dictstrg[name] = name + strgcomp

        # translate the grid
        ## horizontal distance to the component [arcsec]
        xposgridtran = xposgrid[indxpixlelem] - dictchalinpt[dictstrg['xpos']]
        ## vertical distance to the component [arcsec]
        yposgridtran = yposgrid[indxpixlelem] - dictchalinpt[dictstrg['ypos']]
        
        anglgrid = np.sqrt(xposgridtran**2 + yposgridtran**2)
        
        # rotate the grid
        xposgridrttr = np.cos(anglgrid) * xposgridtran - np.sin(anglgrid) * yposgridtran
        yposgridrttr = np.sin(anglgrid) * xposgridtran + np.cos(anglgrid) * yposgridtran
        
        # main halo
        if u == 0:
            axisrati = 1. - dictchalinpt[dictstrg['ellp']]
            facteccc = np.sqrt(1. - axisrati**2)
            factrcor = np.sqrt(axisrati**2 * xposgridrttr**2 + yposgridrttr**2)
            
            deflxposrttr = dictchalinpt[dictstrg['bein']] * axisrati / facteccc *  np.arctan(facteccc * xposgridrttr / factrcor)
            deflyposrttr = dictchalinpt[dictstrg['bein']] * axisrati / facteccc * np.arctanh(facteccc * yposgridrttr / factrcor)
        
        # subhalos
        else:
            # component-centric radius [arcsec]
            radigrid = np.sqrt(xposgridtran**2 + yposgridtran**2)
        
            fracanglasca = radigrid / dictchalinpt['ascasubh'][k]
            
            # second term in the NFW deflection profile
            fact = np.ones_like(fracanglasca)
            indxlowr = np.where(fracanglasca < 1.)[0]
            indxuppr = np.where(fracanglasca > 1.)[0]
            fact[indxlowr] = np.arccosh(1. / fracanglasca[indxlowr]) / np.sqrt(1. - fracanglasca[indxlowr]**2)
            fact[indxuppr] = np.arccos(1. / fracanglasca[indxuppr]) / np.sqrt(fracanglasca[indxuppr]**2 - 1.)
            
            if dictchalinpt['boolasym']:
                factcutf = np.log(fracanglasca / 2.) + fact
            else:
                fracacutasca = acut / asca
                factcutf = fracacutasca**2 / (fracacutasca**2 + 1)**2 * ((fracacutasca**2 + 1. + 2. * (fracanglasca**2 - 1.)) * fact + \
                        np.pi * fracacutasca + (fracacutasca**2 - 1.) * np.log(fracacutasca) + \
                        np.sqrt(fracanglasca**2 + fracacutasca**2) * (-np.pi + (fracacutasca**2 - 1.) / fracacutasca * \
                        np.log(fracanglasca / (np.sqrt(fracanglasca**2 + fracacutasca**2) + fracacutasca))))
            
            deflxposrttr = factcutf * fact * xposgridtran
            deflyposrttr = factcutf * fact * yposgridtran

        # rotate back vector to original basis
        dictchaloutp['deflxposhalo'] = np.cos(anglgrid) * deflxposrttr + np.sin(anglgrid) * deflyposrttr
        dictchaloutp['deflyposhalo'] = -np.sin(anglgrid) * deflxposrttr + np.cos(anglgrid) * deflyposrttr
   
        dictchaloutp['deflhalo'] = np.vstack((dictchaloutp['deflxposhalo'], dictchaloutp['deflyposhalo'])).T
        
        if numbiter > 1 and u == 0:
            dictchaloutp['defltotl'] = np.copy(dictchaloutp['deflhalo'])
        else:
            dictchaloutp['defltotl'] += dictchaloutp['deflsubh'][k]
        

    return dictchaloutp


def retr_caustics(xposgrid, yposgrid, indxpixlelem, dictchalinpt):
   
    from skimage import measure
    
    deflmain = retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt)
    deflsubh = retr_
    magn = retr_magn()
    
    cont = measure.find_contours(magn, 0.8)
    print('cont')
    print(cont)
