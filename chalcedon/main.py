import numpy as np

import tdpy
import aspendos
from tdpy import summgene



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


def eval_modl( \
              dictmodl=None, \
              gdat=None, gdatmodi=None, strgstat=None, strgmodl=None, boolcalcpsfnconv=None, boolcalcpsfnintp=None, \
              ):
    
    if dictmodl is not None:
        pass

    initchro(gdat, gdatmodi, 'elem')

    # grab the sample vector
    indxpara = np.arange(gmodstat.paragenrscalfull.size) 

    if gmod.numbpopl > 0:
        
        # temp -- this may slow down execution
        gmodstat.indxparagenrelemfull = retr_indxparagenrelemfull(gdat, gmodstat.indxelemfull, strgmodl)
        
        # check if all active generative parameters are finite
        if gdat.booldiag:
            indxparatemp = []
            for l in gmod.indxpopl:
                indxparatemp.append(gmodstat.indxparagenrelemfull[l]['full'])
            indxparatemp.append(gmod.indxpara.genrbase)
            gmodstat.indxpara.genrfull = np.concatenate(indxparatemp)
            if not np.isfinite(gmodstat.paragenrscalfull[gmodstat.indxpara.genrfull]).all():
                raise Exception('')

        gmodstat.numbelem = np.empty(gmod.numbpopl, dtype=int)
        indxelem = [[] for l in gmod.indxpopl]
        for l in gmod.indxpopl:
            gmodstat.numbelem[l] = gmodstat.paragenrscalfull[gmod.indxpara.numbelem[l]].astype(int)
            indxelem[l] = np.arange(gmodstat.numbelem[l])
            gmodstat.numbelem[l] = np.sum(gmodstat.numbelem[l])
        
        gmodstat.numbelemtotl = np.sum(gmodstat.numbelem) 

        gmodstat.dictelem = [[] for l in gmod.indxpopl]
        for l in gmod.indxpopl:
            gmodstat.dictelem[l] = dict()
            for strgfeat in gmod.namepara.genrelemdefa:
                gmodstat.dictelem[l][strgfeat] = []
            for nameparagenrelem in gmod.namepara.genrelem[l]:
                gmodstat.dictelem[l][nameparagenrelem] = gmodstat.paragenrscalfull[gmodstat.indxparagenrelemfull[l][nameparagenrelem]]
                if gdat.booldiag:
                    if ((abs(gmodstat.paragenrscalfull[gmodstat.indxparagenrelemfull[l][nameparagenrelem]]) < 1e-100 ) & \
                        (abs(gmodstat.paragenrscalfull[gmodstat.indxparagenrelemfull[l][nameparagenrelem]]) > 0.)).any():
                        raise Exception('')

                    if gmodstat.numbelem[l] != len(gmodstat.dictelem[l][nameparagenrelem]):
                        print('l')
                        print(l)
                        print('gmodstat.numbelem[l]')
                        print(gmodstat.numbelem[l])
                        print('gmodstat.dictelem[l]')
                        print(gmodstat.dictelem[l])
                        print('gmodstat.dictelem[l][nameparagenrelem]')
                        print(gmodstat.dictelem[l][nameparagenrelem])
                        print('nameparagenrelem')
                        print(nameparagenrelem)
                        raise Exception('')
    
        if gdat.boolbinsener:
            if gdat.typeverb > 1:
                print('Calculating element spectra...')
            initchro(gdat, gdatmodi, 'spec')
            for l in gmod.indxpopl:
                if gmod.typeelem[l].startswith('lght'):
                    for strgfeat in gmod.namepara.genrelem[l]:
                        sindcolr = [gmodstat.dictelem[l]['sindcolr%04d' % i] for i in gdat.indxenerinde]
                        gmodstat.dictelem[l]['spec'] = retr_spec(gdat, gmodstat.dictelem[l]['flux'], \
                                                    sind=gmodstat.dictelem[l]['sind'], curv=gmodstat.dictelem[l]['curv'], \
                                                    expc=gmodstat.dictelem[l]['expc'], sindcolr=sindcolr, spectype=gmod.spectype[l])
                        if gmod.typeelem[l].startswith('lghtline'):
                            if gmod.typeelem[l] == 'lghtlinevoig':
                                gmodstat.dictelem[l]['spec'] = retr_spec(gdat, gmodstat.dictelem[l]['flux'], \
                                                                                elin=gmodstat.dictelem[l]['elin'], sigm=gmodstat.dictelem[l]['sigm'], \
                                                                                gamm=gmodstat.dictelem[l]['gamm'], spectype=gmod.spectype[l])
                            else:
                                gmodstat.dictelem[l]['spec'] = retr_spec(gdat, gmodstat.dictelem[l]['flux'], elin=gmodstat.dictelem[l]['elin'], \
                                                                                                edisintp=gdat.edisintp, spectype=gmod.spectype[l])

            stopchro(gdat, gdatmodi, 'spec')
        
        if gdat.booldiag:
            for l in gmod.indxpopl:
                for g, nameparagenrelem in enumerate(gmod.namepara.genrelem[l]):
                    if (gmod.scalpara.genrelem[l][g] != 'gaus' and not gmod.scalpara.genrelem[l][g].startswith('lnor')) and  \
                       (gmod.scalpara.genrelem[l][g] != 'expo' and (gmodstat.dictelem[l][nameparagenrelem] < getattr(gmod.minmpara, nameparagenrelem)).any()) or \
                                        (gmodstat.dictelem[l][nameparagenrelem] > getattr(gmod.maxmpara, nameparagenrelem)).any():
                        
                        print('')
                        print('')
                        print('')
                        print('l, g')
                        print(l, g)
                        print('nameparagenrelem')
                        print(nameparagenrelem)
                        print('gmodstat.dictelem[l][nameparagenrelem]')
                        summgene(gmodstat.dictelem[l][nameparagenrelem])
                        print('getattr(gmod, minm + nameparagenrelem)')
                        print(getattr(gmod.minmpara, nameparagenrelem))
                        print('getattr(gmod, maxm + nameparagenrelem)')
                        print(getattr(gmod.maxmpara, nameparagenrelem))
                        print('gmod.scalpara.genrelem[l][g]')
                        print(gmod.scalpara.genrelem[l][g])
                        raise Exception('')
           
            for l in gmod.indxpopl:
                if gmod.typeelem[l] == 'lens':
                    if gdat.variasca:
                        indx = np.where(gmodstat.paragenrscalfull[gmodstat.indxparagenrelemfull[l]['acut']] < 0.)[0]
                        if indx.size > 0:
                            raise Exception('')
                    if gdat.variacut:
                        indx = np.where(gmodstat.paragenrscalfull[gmodstat.indxparagenrelemfull[l]['asca']] < 0.)[0]
                        if indx.size > 0:
                            raise Exception('')
    
        # calculate element spectra
        for l in gmod.indxpopl:
            if gmod.typeelem[l].startswith('lght'):
                    
                # evaluate horizontal and vertical position for elements whose position is a power law in image-centric radius
                if gmod.typespatdist[l] == 'glc3':
                    gmodstat.dictelem[l]['dlos'], gmodstat.dictelem[l]['xpos'], gmodstat.dictelem[l]['ypos'] = retr_glc3(gmodstat.dictelem[l]['dglc'], \
                                                                                                        gmodstat.dictelem[l]['thet'], gmodstat.dictelem[l]['phii'])
                
                if gmod.typespatdist[l] == 'gangexpo':
                    gmodstat.dictelem[l]['xpos'], gmodstat.dictelem[l]['ypos'], = retr_xposypos(gmodstat.dictelem[l]['gang'], \
                                                                                                        gmodstat.dictelem[l]['aang'])
                    
                    if gdat.booldiag:
                        if gmodstat.numbelem[l] > 0:
                            if np.amin(gmodstat.dictelem[l]['xpos']) < gmod.minmxpos or \
                               np.amax(gmodstat.dictelem[l]['xpos']) > gmod.maxmxpos or \
                               np.amin(gmodstat.dictelem[l]['ypos']) < gmod.minmypos or \
                               np.amax(gmodstat.dictelem[l]['ypos']) > gmod.maxmypos:
                                raise Exception('Bad coordinates!')

                if gmod.typespatdist[l] == 'los3':
                    gmodstat.dictelem[l]['dglc'], gmodstat.dictelem[l]['thet'], gmodstat.dictelem[l]['phii'] = retr_los3(gmodstat.dictelem[l]['dlos'], \
                                                                                                        gmodstat.dictelem[l]['xpos'], gmodstat.dictelem[l]['ypos'])

                # evaluate flux for pulsars
                if gmod.typeelem[l] == 'lghtpntspuls':
                    gmodstat.dictelem[l]['lumi'] = retr_lumipuls(gmodstat.dictelem[l]['geff'], gmodstat.dictelem[l]['magf'], gmodstat.dictelem[l]['per0'])
                if gmod.typeelem[l] == 'lghtpntsagnntrue':
                    gmodstat.dictelem[l]['reds'] = gdat.redsfromdlosobjt(gmodstat.dictelem[l]['dlos'])
                    gmodstat.dictelem[l]['lumi'] = gmodstat.dictelem[l]['lum0'] * (1. + gmodstat.dictelem[l]['reds'])**4
                if gmod.typeelem[l] == 'lghtpntspuls' or gmod.typeelem[l] == 'lghtpntsagnntrue':
                    gmodstat.dictelem[l]['flux'] = retr_flux(gdat, gmodstat.dictelem[l]['lumi'], gmodstat.dictelem[l]['dlos'])
                # evaluate spectra
                if gmod.typeelem[l].startswith('lghtline'):
                    if gmod.typeelem[l] == 'lghtlinevoig':
                        gmodstat.dictelem[l]['spec'] = retr_spec(gdat, gmodstat.dictelem[l]['flux'], elin=gmodstat.dictelem[l]['elin'], sigm=gmodstat.dictelem[l]['sigm'], \
                                                                                                          gamm=gmodstat.dictelem[l]['gamm'], spectype=gmod.spectype[l])
                    else:
                        gmodstat.dictelem[l]['spec'] = retr_spec(gdat, gmodstat.dictelem[l]['flux'], \
                                                                                            elin=gmodstat.dictelem[l]['elin'], edisintp=gdat.edisintp, spectype=gmod.spectype[l])
                else:
                    sindcolr = [gmodstat.dictelem[l]['sindcolr%04d' % i] for i in gdat.indxenerinde]
                    gmodstat.dictelem[l]['spec'] = retr_spec(gdat, gmodstat.dictelem[l]['flux'], sind=gmodstat.dictelem[l]['sind'], curv=gmodstat.dictelem[l]['curv'], \
                                                                                                expc=gmodstat.dictelem[l]['expc'], sindcolr=sindcolr, spectype=gmod.spectype[l])

    stopchro(gdat, gdatmodi, 'elem')
    
    ### evaluate the model
    initchro(gdat, gdatmodi, 'modl')
    
    # process a sample vector and the occupancy list to calculate secondary variables
    if gmod.boollens:
        xpossour = gmodstat.paragenrscalfull[gmod.indxpara.xpossour]
        ypossour = gmodstat.paragenrscalfull[gmod.indxpara.ypossour]
    
        gmodstat.fluxsour = gmodstat.paragenrscalfull[gmod.indxpara.fluxsour]
        if gdat.numbener > 1:
            gmodstat.sindsour = gmodstat.paragenrscalfull[gmod.indxpara.sindsour]
        gmodstat.sizesour = gmodstat.paragenrscalfull[gmod.indxpara.sizesour]
        gmodstat.ellpsour = gmodstat.paragenrscalfull[gmod.indxpara.ellpsour]
        gmodstat.anglsour = gmodstat.paragenrscalfull[gmod.indxpara.anglsour]
    
        gmodstat.beinhost = [[] for e in gmod.indxsersfgrd]
        for e in gmod.indxsersfgrd:
            gmodstat.beinhost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'beinhostisf%d' % e)]
    
        # maybe to be deleted
        #defl = np.zeros((gdat.numbpixlcart, 2))
        
    if gmod.typeemishost != 'none':
        gmodstat.xposhost = [[] for e in gmod.indxsersfgrd]
        gmodstat.yposhost = [[] for e in gmod.indxsersfgrd]
        gmodstat.fluxhost = [[] for e in gmod.indxsersfgrd]
        if gdat.numbener > 1:
            gmodstat.sindhost = [[] for e in gmod.indxsersfgrd]
        gmodstat.sizehost = [[] for e in gmod.indxsersfgrd]
        for e in gmod.indxsersfgrd:
            gmodstat.xposhost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'xposhostisf%d' % e)]
            gmodstat.yposhost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'yposhostisf%d' % e)]
            gmodstat.fluxhost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'fluxhostisf%d' % e)]
            if gdat.numbener > 1:
                gmodstat.sindhost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'sindhostisf%d' % e)]
            gmodstat.sizehost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'sizehostisf%d' % e)]
        gmodstat.ellphost = [[] for e in gmod.indxsersfgrd]
        gmodstat.anglhost = [[] for e in gmod.indxsersfgrd]
        gmodstat.serihost = [[] for e in gmod.indxsersfgrd]
        for e in gmod.indxsersfgrd:
            gmodstat.ellphost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'ellphostisf%d' % e)]
            gmodstat.anglhost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'anglhostisf%d' % e)]
            gmodstat.serihost[e] = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'serihostisf%d' % e)]
    
    # determine the indices of the pixels over which element kernels will be evaluated
    if gdat.boolbindspat:
        if gmod.numbpopl > 0:
            listindxpixlelem = [[] for l in gmod.indxpopl]
            listindxpixlelemconc = [[] for l in gmod.indxpopl]
            for l in gmod.indxpopl:
                if gmodstat.numbelem[l] > 0:
                    listindxpixlelem[l], listindxpixlelemconc[l] = retr_indxpixlelemconc(gdat, strgmodl, gmodstat.dictelem, l)
                    
    if gmod.boollens:
        gmodstat.sherextr = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'sherextr')]
        gmodstat.sangextr = gmodstat.paragenrscalfull[getattr(gmod.indxpara, 'sangextr')]
       
        ## host halo deflection
        initchro(gdat, gdatmodi, 'deflhost')
        deflhost = [[] for e in gmod.indxsersfgrd]
            
        indxpixlmiss = gdat.indxpixlcart

        for e in gmod.indxsersfgrd:
            if gdat.typeverb > 1:
                print('Evaluating the deflection field due to host galaxy %d' % e)
                print('xposhost[e]')
                print(gmodstat.xposhost[e])
                print('yposhost[e]')
                print(gmodstat.yposhost[e])
                print('beinhost[e]')
                print(gmodstat.beinhost[e])
                print('gmodstat.ellphost[e]')
                print(gmodstat.ellphost[e])
                print('gmodstat.anglhost[e]')
                print(gmodstat.anglhost[e])

            deflhost[e] = chalcedon.retr_defl(gdat.xposgrid, gdat.yposgrid, indxpixlmiss, gmodstat.xposhost[e], gmodstat.yposhost[e], \
                                                                        gmodstat.beinhost[e], ellp=gmodstat.ellphost[e], angl=gmodstat.anglhost[e])
             
            if gdat.booldiag:
                if not np.isfinite(deflhost[e]).all():
                    print('')
                    print('')
                    print('')
                    print('gdat.xposgrid')
                    summgene(gdat.xposgrid)
                    print('gdat.yposgrid')
                    summgene(gdat.yposgrid)
                    print('indxpixlmiss')
                    summgene(indxpixlmiss)
                    print('xposhost[e]')
                    print(gmodstat.xposhost[e])
                    print('yposhost[e]')
                    print(gmodstat.yposhost[e])
                    print('beinhost[e]')
                    print(gmodstat.beinhost[e])
                    print('gmodstat.ellphost[e]')
                    print(gmodstat.ellphost[e])
                    print('gmodstat.anglhost[e]')
                    print(gmodstat.anglhost[e])
                    print('deflhost[e]')
                    print(deflhost[e])
                    summgene(deflhost[e])
                    raise Exception('not np.isfinite(deflhost[e]).all()')
        
            if gdat.booldiag:
                indxpixltemp = slice(None)
            
            setattr(gmodstat, 'deflhostisf%d' % e, deflhost[e])
       
            if gdat.typeverb > 1:
                print('deflhost[e]')
                summgene(deflhost[e])
                
            defl += deflhost[e]
            if gdat.typeverb > 1:
                print('After adding the host deflection...')
                print('defl')
                summgene(defl)
        
        stopchro(gdat, gdatmodi, 'deflhost')

        ## external shear
        initchro(gdat, gdatmodi, 'deflextr')
        deflextr = []
        indxpixltemp = gdat.indxpixlcart
        deflextr = retr_deflextr(gdat, indxpixltemp, gmodstat.sherextr, gmodstat.sangextr)
        defl += deflextr
        if gdat.typeverb > 1:
            print('After adding the external deflection...')
            print('defl')
            summgene(defl)
        stopchro(gdat, gdatmodi, 'deflextr')
    
    if gmod.boolneedpsfnconv:
        
        initchro(gdat, gdatmodi, 'psfnconv')
        
        # compute the PSF convolution object
        if boolcalcpsfnconv:
            psfnconv = [[[] for i in gdat.indxener] for m in gdat.indxdqlt]
            gmodstat.psfn = retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.angl, gmod.typemodlpsfn, strgmodl)
            fwhm = 2. * retr_psfnwdth(gdat, gmodstat.psfn, 0.5)
            for mm, m in enumerate(gdat.indxdqlt):
                for ii, i in enumerate(gdat.indxener):
                    if gmod.typemodlpsfn == 'singgaus':
                        sigm = psfp[i+m*gdat.numbener]
                    else:
                        sigm = fwhm[i, m] / 2.355
                    psfnconv[mm][ii] = AiryDisk2DKernel(sigm / gdat.sizepixl)
            
        stopchro(gdat, gdatmodi, 'psfnconv')
    
    if gmod.boolneedpsfnintp:
        
        # compute the PSF interpolation object
        if boolcalcpsfnintp:
            if gdat.typepixl == 'heal':
                gmodstat.psfn = retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.angl, gmod.typemodlpsfn, strgmodl)
                gmodstat.psfnintp = sp.interpolate.interp1d(gdat.blimpara.angl, gmodstat.psfn, axis=1, fill_value='extrapolate')
                fwhm = 2. * retr_psfnwdth(gdat, gmodstat.psfn, 0.5)
            
            elif gdat.typepixl == 'cart':
                if gdat.kernevaltype == 'ulip':
                    gmodstat.psfn = retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.angl, gmod.typemodlpsfn, strgmodl)
                    gmodstat.psfnintp = sp.interpolate.interp1d(gdat.blimpara.angl, gmodstat.psfn, axis=1, fill_value='extrapolate')

                if gdat.kernevaltype == 'bspx':
                    
                    gmodstat.psfn = retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.anglcart.flatten(), gmod.typemodlpsfn, strgmodl)
                    
                    # side length of the upsampled kernel
                    gdat.numbsidekernusam = 100
                    # side length of the original kernel
                    gdat.numbsidekern = gdat.numbsidekernusam / factkernusam 
                    gdat.indxsidekern = np.arange(gdat.numbsidekern)

    	        	# pad by one row and one column
    	        	#psf = np.zeros((gdat.numbsidekernusam+1, gdat.numbsidekernusam+1))
    	        	#psf[0:gdat.numbsidekernusam, 0:gdat.numbsidekernusam] = psf0
		        	
    	        	# make design matrix for each factkernusam x factkernusam region
                    nx = factkernusam + 1
                    y, x = mgrid[0:nx, 0:nx] / float(factkernusam)
                    x = x.flatten()
                    y = y.flatten()
                    kernmatrdesi = np.array([full(nx*nx, 1), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).T
    	        	
                    # output np.array of coefficients
                    gmodstat.psfnintp = np.empty((gdat.numbsidekern, gdat.numbsidekern, kernmatrdesi.shape[1]))

    	        	# solve p = kernmatrdesi psfnintp for psfnintp
                    for iy in gdat.indxsidekern:
                        for ix in gdat.indxsidekern:
                            p = psf[iy*factkernusam:(iy+1)*factkernusam+1, ix*factkernusam:(ix+1)*factkernusam+1].flatten()
                            gmodstat.psfnintp[iy, ix, :] = dot(linalg.inv(dot(kernmatrdesi.T, kernmatrdesi)), dot(kernmatrdesi.T, p))
        else:
            gmodstat.psfnintp = gdat.fitt.this.psfnintp
    
        if gdat.booldiag:
            if not np.isfinite(gmodstat.psfnintp(0.05)).all():
                print('')
                print('')
                print('')
                raise Exception('')
    
    if gmod.numbpopl > 0:
        if gmod.boolelemdeflsubhanyy:
            deflsubh = np.zeros((gdat.numbpixl, 2))
    
        if gmod.boolelemdeflsubhanyy:
            initchro(gdat, gdatmodi, 'elemdeflsubh')
            if gdat.typeverb > 1:
                print('Perturbing subhalo deflection field')
            for l in gmod.indxpopl:
                if gmod.typeelem[l] == 'lens':
                    for kk, k in enumerate(indxelem[l]):
                        asca = gmodstat.dictelem[l]['asca'][k]
                        acut = gmodstat.dictelem[l]['acut'][k]
                        if gmod.typeelemspateval[l] == 'locl':
                            indxpixl = listindxpixlelem[l][kk]
                        else:
                            indxpixl = gdat.indxpixl
                        deflsubh[indxpixl, :] += chalcedon.retr_defl(gdat.xposgrid, gdat.yposgrid, indxpixl, \
                                                     gmodstat.dictelem[l]['xpos'][kk], gmodstat.dictelem[l]['ypos'][kk], gmodstat.dictelem[l]['defs'][kk], \
                                                     asca=asca, acut=acut)
            
                    # temp -- find out what is causing the features in the element convergence maps
                    #for kk, k in enumerate(indxelem[l]):
                    #    indxpixlpnts = retr_indxpixl(gdat, gmodstat.dictelem[l]['ypos'][kk], gmodstat.dictelem[l]['xpos'][kk])
                    #    if deflsubh[listindxpixlelem[l][kk], :]
            
            setattr(gmodstat, 'deflsubh', deflsubh)
            
            if gdat.booldiag:
                if not np.isfinite(deflsubh).all():
                    raise Exception('Element deflection is not finite.')

            defl += deflsubh
            if gdat.typeverb > 1:
                print('After adding subhalo deflection to the total deflection')
                print('defl')
                summgene(defl)

            stopchro(gdat, gdatmodi, 'elemdeflsubh')

    # evaluate surface brightnesses
    sbrt = dict()
    for name in gmod.listnamediff:
        sbrt[name] = []
    
    ## due to elements
    if gmod.numbpopl > 0:
        if gmod.boolelemsbrtdfncanyy:
            sbrtdfnc = np.zeros_like(gdat.expo)
        if gmod.boolelemsbrtextsbgrdanyy: 
            sbrtextsbgrd = np.zeros_like(gdat.expo)
        
        # element kernel evaluation
        if gmod.boolelemsbrtdfncanyy:
            initchro(gdat, gdatmodi, 'elemsbrtdfnc')
            sbrt['dfnc'] = []
            for l in gmod.indxpopl:
                if gmod.boolelemsbrtdfnc[l]:
                    for k in range(gmodstat.numbelem[l]):
                        if gmod.boolelemlght[l]:
                            varbamplextd = gmodstat.dictelem[l]['spec'][:, k]
                        if gmod.typeelem[l].startswith('clus'):
                            varbamplextd = gmodstat.dictelem[l]['nobj'][None, k]
                        if gmod.typeelem[l] == 'clusvari':
                            sbrtdfnc[0, listindxpixlelem[l][k], 0] += gmodstat.dictelem[l]['nobj'][k] / 2. / np.pi / gmodstat.dictelem[l]['gwdt'][k]**2 * \
                                np.exp(-0.5 * ((gmodstat.dictelem[l]['xpos'][k] - gdat.xposgrid[listindxpixlelem[l][k]])**2 + \
                                    (gmodstat.dictelem[l]['ypos'][k] - gdat.yposgrid[listindxpixlelem[l][k]])**2) / gmodstat.dictelem[l]['gwdt'][k]**2)
                            
                        if gmod.boolelempsfn[l]:
                            sbrtdfnc[:, listindxpixlelem[l][k], :] += retr_sbrtpnts(gdat, gmodstat.dictelem[l]['xpos'][k], \
                                                             gmodstat.dictelem[l]['ypos'][k], varbamplextd, gmodstat.psfnintp, listindxpixlelem[l][k])
                        
                        if gmod.typeelem[l].startswith('lghtline'):
                            sbrtdfnc[:, 0, 0] += gmodstat.dictelem[l]['spec'][:, k]
                        
            sbrt['dfnc'] = sbrtdfnc
            
            setattr(gmodstat, 'sbrtdfnc', sbrt['dfnc'])
            stopchro(gdat, gdatmodi, 'elemsbrtdfnc')
            
            if gdat.booldiag:
                if not np.isfinite(sbrtdfnc).all():
                    raise Exception('Element delta function brightness not finite.')

                cntppntschec = retr_cntp(gdat, sbrt['dfnc'])
                numbelemtemp = 0
                for l in gmod.indxpopl:
                    if gmod.boolelemsbrtdfnc[l]:
                        numbelemtemp += np.sum(gmodstat.numbelem[l])
                if np.amin(cntppntschec) < -0.1:
                    raise Exception('Point source spectral surface brightness is not positive-definite.')
            
        
        if gmod.boolelemsbrtextsbgrdanyy:
            initchro(gdat, gdatmodi, 'elemsbrtextsbgrd')
            if strgstat == 'this':
                for l in gmod.indxpopl:
                    if gmod.typeelem[l] == 'lghtgausbgrd':
                        for k in range(gmodstat.numbelem[l]):
                            sbrtextsbgrd[:, listindxpixlelem[l][k], :] += gmodstat.dictelem[l]['spec'][:, k, None, None] / \
                                    2. / np.pi / gmodstat.dictelem[l]['gwdt'][k]**2 * \
                                    np.exp(-0.5 * ((gmodstat.dictelem[l]['xpos'][k] - gdat.xposgrid[None, listindxpixlelem[l][k], None])**2 + \
                                    (gmodstat.dictelem[l]['ypos'][k] - gdat.yposgrid[None, listindxpixlelem[l][k], None])**2) / gmodstat.dictelem[l]['gwdt'][k]**2)
                
                setattr(gmodstat, 'sbrtextsbgrd', sbrtextsbgrd)
            sbrt['extsbgrd'] = []
            sbrt['extsbgrd'] = sbrtextsbgrd
            stopchro(gdat, gdatmodi, 'elemsbrtextsbgrd')
            
            if gdat.booldiag:
                cntppntschec = retr_cntp(gdat, sbrt['extsbgrd'])
                if np.amin(cntppntschec) < -0.1:
                    raise Exception('Point source spectral surface brightness is not positive-definite.')
        
    
    ## lensed surface brightness
    if gmod.boollens:
        
        initchro(gdat, gdatmodi, 'sbrtlens')
        
        if gdat.typeverb > 1:
            print('Evaluating lensed surface brightness...')
        
        if strgstat == 'this' or gmod.numbpopl > 0 and gmod.boolelemsbrtextsbgrdanyy:
            sbrt['bgrd'] = []
        if gmod.numbpopl > 0 and gmod.boolelemsbrtextsbgrdanyy:
            sbrt['bgrdgalx'] = []
        
        if gdat.numbener > 1:
            specsour = retr_spec(gdat, np.array([gmodstat.fluxsour]), sind=np.array([gmodstat.sindsour]))
        else:
            specsour = np.array([gmodstat.fluxsour])
        
        if gmod.numbpopl > 0 and gmod.boolelemsbrtextsbgrdanyy:
        
            if gdat.typeverb > 1:
                print('Interpolating the background emission...')

            sbrt['bgrdgalx'] = retr_sbrtsers(gdat, gdat.xposgrid[indxpixlelem[0]], gdat.yposgrid[indxpixlelem[0]], \
                                                                            xpossour, ypossour, specsour, gmodstat.sizesour, gmodstat.ellpsour, gmodstat.anglsour)
            
            sbrt['bgrd'] = sbrt['bgrdgalx'] + sbrtextsbgrd
        
            sbrt['lens'] = np.empty_like(gdat.cntpdata)
            for ii, i in enumerate(gdat.indxener):
                for mm, m in enumerate(gdat.indxdqlt):
                    sbrtbgrdobjt = sp.interpolate.RectBivariateSpline(gdat.bctrpara.yposcart, gdat.bctrpara.xposcart, \
                                                            sbrt['bgrd'][ii, :, mm].reshape((gdat.numbsidecart, gdat.numbsidecart)).T)
                    
                    yposprim = gdat.yposgrid[indxpixlelem[0]] - defl[indxpixlelem[0], 1]
                    xposprim = gdat.xposgrid[indxpixlelem[0]] - defl[indxpixlelem[0], 0]
                    # temp -- T?
                    sbrt['lens'][ii, :, m] = sbrtbgrdobjt(yposprim, xposprim, grid=False).flatten()
        else:
            if gdat.typeverb > 1:
                print('Not interpolating the background emission...')
            
            sbrt['lens'] = retr_sbrtsers(gdat, gdat.xposgrid - defl[gdat.indxpixl, 0], \
                                                   gdat.yposgrid - defl[gdat.indxpixl, 1], \
                                                   xpossour, ypossour, specsour, gmodstat.sizesour, gmodstat.ellpsour, gmodstat.anglsour)
            
            sbrt['bgrd'] = retr_sbrtsers(gdat, gdat.xposgrid, \
                                                   gdat.yposgrid, \
                                                   xpossour, ypossour, specsour, gmodstat.sizesour, gmodstat.ellpsour, gmodstat.anglsour)
            
        setattr(gmodstat, 'sbrtlens', sbrt['lens'])

        if gdat.booldiag:
            if not np.isfinite(sbrt['lens']).all():
                raise Exception('Lensed emission is not finite.')
            if (sbrt['lens'] == 0).all():
                raise Exception('Lensed emission is zero everywhere.')

        stopchro(gdat, gdatmodi, 'sbrtlens')
        
    ## host galaxy
    if gmod.typeemishost != 'none':
        initchro(gdat, gdatmodi, 'sbrthost')

        for e in gmod.indxsersfgrd:
            if gdat.typeverb > 1:
                print('Evaluating the host galaxy surface brightness...')
            
            if gdat.numbener > 1:
                spechost = retr_spec(gdat, np.array([gmodstat.fluxhost[e]]), sind=np.array([gmodstat.sindhost[e]]))
            else:
                spechost = np.array([gmodstat.fluxhost[e]])
            
            sbrt['hostisf%d' % e] = retr_sbrtsers(gdat, gdat.xposgrid, gdat.yposgrid, gmodstat.xposhost[e], \
                                                         gmodstat.yposhost[e], spechost, gmodstat.sizehost[e], gmodstat.ellphost[e], gmodstat.anglhost[e], gmodstat.serihost[e])
            
            setattr(gmodstat, 'sbrthostisf%d' % e, sbrt['hostisf%d' % e])
                
        stopchro(gdat, gdatmodi, 'sbrthost')
    
    ## total model
    initchro(gdat, gdatmodi, 'sbrtmodl')
    if gdat.typeverb > 1:
        print('Summing up the model emission...')
    
    sbrt['modlraww'] = np.zeros((gdat.numbener, gdat.numbpixlcart, gdat.numbdqlt))
    for name in gmod.listnamediff:
        if name.startswith('back'):
            gmod.indxbacktemp = int(name[4:8])
            
            if gdat.typepixl == 'heal' and (gmod.typeevalpsfn == 'full' or gmod.typeevalpsfn == 'conv') and not gmod.boolunifback[gmod.indxbacktemp]:
                sbrttemp = getattr(gmod, 'sbrtbackhealfull')[gmod.indxbacktemp]
            else:
                sbrttemp = gmod.sbrtbacknorm[gmod.indxbacktemp]
           
            if gmod.boolspecback[gmod.indxbacktemp]:
                sbrt[name] = sbrttemp * bacp[gmod.indxbacpback[gmod.indxbacktemp]]
            else:
                sbrt[name] = sbrttemp * bacp[gmod.indxbacpback[gmod.indxbacktemp][gdat.indxener]][:, None, None]
        
        sbrt['modlraww'] += sbrt[name]
        
        if gdat.booldiag:
            if np.amax(sbrttemp) == 0.:
                raise Exception('')

    # convolve the model with the PSF
    if gmod.convdiffanyy and (gmod.typeevalpsfn == 'full' or gmod.typeevalpsfn == 'conv'):
        sbrt['modlconv'] = []
        # temp -- isotropic background proposals are unnecessarily entering this clause
        if gdat.typeverb > 1:
            print('Convolving the model image with the PSF...') 
        sbrt['modlconv'] = np.zeros((gdat.numbener, gdat.numbpixl, gdat.numbdqlt))
        for ii, i in enumerate(gdat.indxener):
            for mm, m in enumerate(gdat.indxdqlt):
                if gdat.strgcnfg == 'pcat_ferm_igal_simu_test':
                    print('Convolving ii, i, mm, m')
                    print(ii, i, mm, m)
                if gdat.typepixl == 'cart':
                    if gdat.numbpixl == gdat.numbpixlcart:
                        sbrt['modlconv'][ii, :, mm] = convolve_fft(sbrt['modlraww'][ii, :, mm].reshape((gdat.numbsidecart, gdat.numbsidecart)), \
                                                                                                                             psfnconv[mm][ii]).flatten()
                    else:
                        sbrtfull = np.zeros(gdat.numbpixlcart)
                        sbrtfull[gdat.indxpixlrofi] = sbrt['modlraww'][ii, :, mm]
                        sbrtfull = sbrtfull.reshape((gdat.numbsidecart, gdat.numbsidecart))
                        sbrt['modlconv'][ii, :, mm] = convolve_fft(sbrtfull, psfnconv[mm][ii]).flatten()[gdat.indxpixlrofi]
                    indx = np.where(sbrt['modlconv'][ii, :, mm] < 1e-50)
                    sbrt['modlconv'][ii, indx, mm] = 1e-50
                if gdat.typepixl == 'heal':
                    sbrt['modlconv'][ii, :, mm] = hp.smoothing(sbrt['modlraww'][ii, :, mm], fwhm=fwhm[i, m])[gdat.indxpixlrofi]
                    sbrt['modlconv'][ii, :, mm][np.where(sbrt['modlraww'][ii, :, mm] <= 1e-50)] = 1e-50
        
        setattr(gmodstat, 'sbrtmodlconv', sbrt['modlconv'])
        # temp -- this could be made faster -- need the copy() statement because sbrtdfnc gets added to sbrtmodl afterwards
        sbrt['modl'] = np.copy(sbrt['modlconv'])
    else:
        if gdat.typeverb > 1:
            print('Skipping PSF convolution of the model...')
        sbrt['modl'] = np.copy(sbrt['modlraww'])
    
    if gdat.typeverb > 1:
        print('sbrt[modl]')
        summgene(sbrt['modl'])

    ## add PSF-convolved delta functions to the model
    if gmod.numbpopl > 0 and gmod.boolelemsbrtdfncanyy:
        sbrt['modl'] += sbrt['dfnc']
    stopchro(gdat, gdatmodi, 'sbrtmodl')
    
    if gdat.typeverb > 1:
        print('sbrt[modl]')
        summgene(sbrt['modl'])



