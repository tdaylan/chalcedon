

def retr_mdencrit(gdat, adissour, adishost, adishostsour):
    
    mdencrit = gdat.factnewtlght / 4. / np.pi * adissour / adishostsour / adishost
        
    return mdencrit


def retr_massfrombein(gdat, adissour, adishost, adishostsour):

    mdencrit = retr_mdencrit(gdat, adissour, adishost, adishostsour)
    massfrombein = np.pi * adishost**2 * mdencrit

    return massfrombein


def retr_factmcutfromdefs(gdat, adissour, adishost, adishostsour, asca, acut):
    
    mdencrit = retr_mdencrit(gdat, adissour, adishost, adishostsour)
    
    fracacutasca = acut / asca
    
    factmcutfromdefs = np.pi * adishost**2 * mdencrit * asca * retr_mcutfrommscl(fracacutasca)

    return factmcutfromdefs


def retr_mcut(gdat, defs, asca, acut, adishost, mdencrit):
    
    mscl = defs * np.pi * adishost**2 * mdencrit * asca
    fracacutasca = acut / asca
    mcut = mscl * retr_mcutfrommscl(fracacutasca)
    
    return mcut


def retr_mcutfrommscl(fracacutasca):
    
    mcut = fracacutasca**2 / (fracacutasca**2 + 1.)**2 * ((fracacutasca**2 - 1.) * np.log(fracacutasca) + fracacutasca * np.pi - (fracacutasca**2 + 1.))

    return mcut


def retr_defl(gdat, indxpixlelem, lgal, bgal, angllens, ellp=None, angl=None, rcor=None, asca=None, acut=None):
    
    # translate the grid
    lgaltran = gdat.lgalgrid[indxpixlelem] - lgal
    bgaltran = gdat.bgalgrid[indxpixlelem] - bgal
    
    if acut is not None:
        defs = angllens
        angl = np.sqrt(lgaltran**2 + bgaltran**2)
        defl = retr_deflcutf(angl, defs, asca, acut)
        defllgal = lgaltran / angl * defl
        deflbgal = bgaltran / angl * defl

    else:
        bein = angllens

        # rotate the grid
        lgalrttr = np.cos(angl) * lgaltran - np.sin(angl) * bgaltran
        bgalrttr = np.sin(angl) * lgaltran + np.cos(angl) * bgaltran
        
        axisrati = 1. - ellp
        facteccc = np.sqrt(1. - axisrati**2)
        factrcor = np.sqrt(axisrati**2 * lgalrttr**2 + bgalrttr**2)
        defllgalrttr = bein * axisrati / facteccc *  np.arctan(facteccc * lgalrttr / factrcor)
        deflbgalrttr = bein * axisrati / facteccc * np.arctanh(facteccc * bgalrttr / factrcor)
        
        # totate back vector to original basis
        defllgal = np.cos(angl) * defllgalrttr + np.sin(angl) * deflbgalrttr
        deflbgal = -np.sin(angl) * defllgalrttr + np.cos(angl) * deflbgalrttr
   
    defl = np.vstack((defllgal, deflbgal)).T
    
    return defl


def eval_modl():
    pass

