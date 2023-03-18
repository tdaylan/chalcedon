import numpy as np

import tdpy


def retr_factmcutfromdefs(adissour, adislens, adislenssour, asca, acut):
    
    mdencrit = retr_mdencrit(adissour, adislens, adislenssour)
    
    fracacutasca = acut / asca
    
    factmcutfromdefs = np.pi * adislens**2 * mdencrit * asca * retr_mcutfrommscl(fracacutasca)

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


def retr_deflcutf(angl, defs, asca, acut, asym=False):

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
    
    # translate the grid
    xpostran = xposgrid[indxpixlelem] - xpos
    ypostran = yposgrid[indxpixlelem] - ypos
    
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


def eval_modl(srbtbkgd, masslens, xposlens, yposlens):
    pass

