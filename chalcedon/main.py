import numpy as np

import time as modutime

import astropy
import astropy.convolution

import tdpy
import aspendos
from tdpy import summgene
import nicomedia


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


def eval_emislens( \
                  # input dictionary
                  dictchalinpt=None, \
                  
                  # number of pixels on a side
                  numbsidecart=None, \

                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1, \
                  
                  # Boolean flag to turn on diagnostic mode
                  booldiag=True, \
                 ):
    
    '''
    Calculate the emission due to graviationally lensed-sources background sources
    '''
    timeinit = modutime.time()
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    dictinpt = dict(locals())
    for attr, valu in dictinpt.items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    if dictchalinpt is not None:
        pass
    
    # output dictionary
    dictchaloutp = dict()
    
    #objttimeprof = tdpt.retr_objttimeprof('emislens')

    #objttimeprof.initchro(gdat, gdatmodi, 'elem')
    # grab the sample vector
    #indxpara = np.arange(paragenr.size) 
    
    numbpopl = 1
    
    if not 'listnamesersfgrd' in dictchalinpt:
        dictchalinpt['listnamesersfgrd'] = []

    if not 'typeemishost' in dictchalinpt:
        dictchalinpt['typeemishost'] = 'Sersic'

    if not 'numbener' in dictchalinpt:
        dictchalinpt['numbener'] = 1

    # process a sample vector and the occupancy list to calculate secondary variables

    numbsersfgrd = len(dictchalinpt['listnamesersfgrd'])
    indxsersfgrd = np.arange(numbsersfgrd)
    beinhost = [[] for e in indxsersfgrd]
    for e in indxsersfgrd:
        beinhost[e] = paragenr[getattr(indxpara, 'beinhostisf%d' % e)]
    
    # maybe to be deleted
    #if dictchalinpt['typeemishost'] != 'none':
    #    xposhost = [[] for e in indxsersfgrd]
    #    yposhost = [[] for e in indxsersfgrd]
    #    fluxhost = [[] for e in indxsersfgrd]
    #    if dictchalinpt['numbener'] > 1:
    #        sindhost = [[] for e in indxsersfgrd]
    #    sizehost = [[] for e in indxsersfgrd]
    #    for e in indxsersfgrd:
    #        xposhost[e] = paragenr[getattr(indxpara, 'xposhostisf%d' % e)]
    #        yposhost[e] = paragenr[getattr(indxpara, 'yposhostisf%d' % e)]
    #        fluxhost[e] = paragenr[getattr(indxpara, 'fluxhostisf%d' % e)]
    #        if dictchalinpt['numbener'] > 1:
    #            sindhost[e] = paragenr[getattr(indxpara, 'sindhostisf%d' % e)]
    #        sizehost[e] = paragenr[getattr(indxpara, 'sizehostisf%d' % e)]
    #    ellphost = [[] for e in indxsersfgrd]
    #    anglhost = [[] for e in indxsersfgrd]
    #    serihost = [[] for e in indxsersfgrd]
    #    for e in indxsersfgrd:
    #        ellphost[e] = paragenr[getattr(indxpara, 'ellphostisf%d' % e)]
    #        anglhost[e] = paragenr[getattr(indxpara, 'anglhostisf%d' % e)]
    #        serihost[e] = paragenr[getattr(indxpara, 'serihostisf%d' % e)]
    
    ## host halo deflection
    #objttimeprof.initchro(gdat, gdatmodi, 'deflhost')
    deflhost = [[] for e in indxsersfgrd]
    
    if numbsidecart is None:
        numbsidecart = 100
        maxmfovw = 2.
        numbpixlcart = numbsidecart**2
        indxpixlcart = np.arange(numbpixlcart)
        xposside = np.linspace(-maxmfovw, maxmfovw, 100)
        yposside = xposside
        xposgrid, yposgrid = np.meshgrid(xposside, yposside, indexing='ij')
        xposgridflat = xposgrid.flatten()
        yposgridflat = yposgrid.flatten()

    defl = np.zeros((numbpixlcart, 2))
        
    for e in indxsersfgrd:

        deflhost[e] = retr_defl(gdat.xposgrid, gdat.yposgrid, indxpixlcart, xposhost[e], yposhost[e], \
                                                                    beinhost[e], ellp=ellphost[e], angl=anglhost[e])
         
        if gdat.booldiag:
            if not np.isfinite(deflhost[e]).all():
                print('')
                print('')
                print('')
                raise Exception('not np.isfinite(deflhost[e]).all()')
    
        if gdat.booldiag:
            indxpixltemp = slice(None)
        
        setattr(gmodstat, 'deflhostisf%d' % e, deflhost[e])
    
        if typeverb > 1:
            print('deflhost[e]')
            summgene(deflhost[e])
            
        defl += deflhost[e]
        if typeverb > 1:
            print('After adding the host deflection...')
            print('defl')
            summgene(defl)
    
    #objttimeprof.stopchro(gdat, gdatmodi, 'deflhost')

    ## external shear
    #objttimeprof.initchro(gdat, gdatmodi, 'deflextr')
    deflextr = retr_deflextr(xposgridflat, yposgridflat, dictchalinpt['sherextr'], dictchalinpt['sangextr'])
    defl += deflextr
    
    if typeverb > 1:
        print('After adding the external deflection...')
        print('defl')
        summgene(defl)
    
    #objttimeprof.stopchro(gdat, gdatmodi, 'deflextr')
    
    typeevalpsfn = 'full'

    boolneedpsfnconv = typeevalpsfn == 'conv' or typeevalpsfn == 'full'
    boolcalcpsfnconv = boolneedpsfnconv
    
    sizepixl = 0.11 # [arcsec]

    numbdqlt = 1
    indxdqlt = np.arange(numbdqlt)
    numbener = 1
    indxener = np.arange(numbener)
    
    arryangl = np.linspace(0.1, 2., 100)
    
    typemodlpsfn = 'singgaus'
    
    dictpara = dict()

    dictpara['sigc'] = np.array([[1.]])

    if boolneedpsfnconv:
        
        #objttimeprof.initchro(gdat, gdatmodi, 'psfnconv')
        
        # compute the PSF convolution object
        if boolcalcpsfnconv:
            objtpsfnconv = [[[] for i in indxener] for m in indxdqlt]
            psfn = nicomedia.retr_psfn(arryangl, dictpara, indxener, typemodlpsfn)
            fwhm = 2. * nicomedia.retr_psfnwdth(psfn, arryangl, 0.5)
            for mm, m in enumerate(indxdqlt):
                for ii, i in enumerate(indxener):
                    if typemodlpsfn == 'singgaus':
                        sigm = dictpara['sigc'][i, m]
                    else:
                        sigm = fwhm[i, m] / 2.355
                    objtpsfnconv[mm][ii] =  astropy.convolution.AiryDisk2DKernel(sigm / sizepixl)
            
        #objttimeprof.stopchro(gdat, gdatmodi, 'psfnconv')
    
    if boolneedpsfnintp:
        
        # compute the PSF interpolation object
        if boolcalcpsfnintp:
            if gdat.typepixl == 'heal':
                psfn = nicomedia.retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.angl, typemodlpsfn, strgmodl)
                psfnintp = sp.interpolate.interp1d(gdat.blimpara.angl, psfn, axis=1, fill_value='extrapolate')
                fwhm = 2. * nicomedia.retr_psfnwdth(gdat, arryangl, psfn, 0.5)
            
            elif gdat.typepixl == 'cart':
                if gdat.kernevaltype == 'ulip':
                    psfn = nicomedia.retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.angl, typemodlpsfn, strgmodl)
                    psfnintp = sp.interpolate.interp1d(gdat.blimpara.angl, psfn, axis=1, fill_value='extrapolate')

                if gdat.kernevaltype == 'bspx':
                    
                    psfn = nicomedia.retr_psfn(gdat, psfp, gdat.indxener, gdat.blimpara.anglcart.flatten(), typemodlpsfn, strgmodl)
                    
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
                    psfnintp = np.empty((gdat.numbsidekern, gdat.numbsidekern, kernmatrdesi.shape[1]))

    	        	# solve p = kernmatrdesi psfnintp for psfnintp
                    for iy in gdat.indxsidekern:
                        for ix in gdat.indxsidekern:
                            p = psf[iy*factkernusam:(iy+1)*factkernusam+1, ix*factkernusam:(ix+1)*factkernusam+1].flatten()
                            psfnintp[iy, ix, :] = dot(linalg.inv(dot(kernmatrdesi.T, kernmatrdesi)), dot(kernmatrdesi.T, p))
        else:
            psfnintp = gdat.fitt.this.psfnintp
    
        if gdat.booldiag:
            if not np.isfinite(psfnintp(0.05)).all():
                print('')
                print('')
                print('')
                raise Exception('')
    
    if numbpopl > 0:
        if boolelemdeflsubhanyy:
            deflsubh = np.zeros((gdat.numbpixl, 2))
    
        if boolelemdeflsubhanyy:
            
            #objttimeprof.initchro(gdat, gdatmodi, 'elemdeflsubh')
            
            if typeverb > 1:
                print('Perturbing subhalo deflection field')
            for l in indxpopl:
                if typeelem[l] == 'lens':
                    for kk, k in enumerate(indxelem[l]):
                        asca = dictelem[l]['asca'][k]
                        acut = dictelem[l]['acut'][k]
                        if typeelemspateval[l] == 'locl':
                            indxpixl = listindxpixlelem[l][kk]
                        else:
                            indxpixl = gdat.indxpixl
                        deflsubh[indxpixl, :] += chalcedon.retr_defl(gdat.xposgrid, gdat.yposgrid, indxpixl, \
                                                     dictelem[l]['xpos'][kk], dictelem[l]['ypos'][kk], dictelem[l]['defs'][kk], \
                                                     asca=asca, acut=acut)
            
                    # temp -- find out what is causing the features in the element convergence maps
                    #for kk, k in enumerate(indxelem[l]):
                    #    indxpixlpnts = retr_indxpixl(gdat, dictelem[l]['ypos'][kk], dictelem[l]['xpos'][kk])
                    #    if deflsubh[listindxpixlelem[l][kk], :]
            
            setattr(gmodstat, 'deflsubh', deflsubh)
            
            if gdat.booldiag:
                if not np.isfinite(deflsubh).all():
                    raise Exception('Element deflection is not finite.')

            defl += deflsubh
            if typeverb > 1:
                print('After adding subhalo deflection to the total deflection')
                print('defl')
                summgene(defl)

            
            #objttimeprof.stopchro(gdat, gdatmodi, 'elemdeflsubh')

    # evaluate surface brightnesses
    sbrt = dict()
    for name in listnamediff:
        sbrt[name] = []
    
    ## due to elements
    if numbpopl > 0:
        if boolelemsbrtdfncanyy:
            sbrtdfnc = np.zeros_like(gdat.expo)
        if boolelemsbrtextsbgrdanyy: 
            sbrtextsbgrd = np.zeros_like(gdat.expo)
        
        # element kernel evaluation
        if boolelemsbrtdfncanyy:
            #objttimeprof.initchro(gdat, gdatmodi, 'elemsbrtdfnc')
            sbrt['dfnc'] = []
            for l in indxpopl:
                if boolelemsbrtdfnc[l]:
                    for k in range(numbelem[l]):
                        if boolelemlght[l]:
                            varbamplextd = dictelem[l]['spec'][:, k]
                        if typeelem[l].startswith('clus'):
                            varbamplextd = dictelem[l]['nobj'][None, k]
                        if typeelem[l] == 'clusvari':
                            sbrtdfnc[0, listindxpixlelem[l][k], 0] += dictelem[l]['nobj'][k] / 2. / np.pi / dictelem[l]['gwdt'][k]**2 * \
                                np.exp(-0.5 * ((dictelem[l]['xpos'][k] - gdat.xposgrid[listindxpixlelem[l][k]])**2 + \
                                    (dictelem[l]['ypos'][k] - gdat.yposgrid[listindxpixlelem[l][k]])**2) / dictelem[l]['gwdt'][k]**2)
                            
                        if boolelempsfn[l]:
                            sbrtdfnc[:, listindxpixlelem[l][k], :] += retr_sbrtpnts(gdat, dictelem[l]['xpos'][k], \
                                                             dictelem[l]['ypos'][k], varbamplextd, psfnintp, listindxpixlelem[l][k])
                        
                        if typeelem[l].startswith('lghtline'):
                            sbrtdfnc[:, 0, 0] += dictelem[l]['spec'][:, k]
                        
            sbrt['dfnc'] = sbrtdfnc
            
            setattr(gmodstat, 'sbrtdfnc', sbrt['dfnc'])
            #objttimeprof.stopchro(gdat, gdatmodi, 'elemsbrtdfnc')
            
            if gdat.booldiag:
                if not np.isfinite(sbrtdfnc).all():
                    raise Exception('Element delta function brightness not finite.')

                cntppntschec = retr_cntp(gdat, sbrt['dfnc'])
                numbelemtemp = 0
                for l in indxpopl:
                    if boolelemsbrtdfnc[l]:
                        numbelemtemp += np.sum(numbelem[l])
                if np.amin(cntppntschec) < -0.1:
                    raise Exception('Point source spectral surface brightness is not positive-definite.')
            
        
        if boolelemsbrtextsbgrdanyy:
            #objttimeprof.initchro(gdat, gdatmodi, 'elemsbrtextsbgrd')
            if strgstat == 'this':
                for l in indxpopl:
                    if typeelem[l] == 'lghtgausbgrd':
                        for k in range(numbelem[l]):
                            sbrtextsbgrd[:, listindxpixlelem[l][k], :] += dictelem[l]['spec'][:, k, None, None] / \
                                    2. / np.pi / dictelem[l]['gwdt'][k]**2 * \
                                    np.exp(-0.5 * ((dictelem[l]['xpos'][k] - gdat.xposgrid[None, listindxpixlelem[l][k], None])**2 + \
                                    (dictelem[l]['ypos'][k] - gdat.yposgrid[None, listindxpixlelem[l][k], None])**2) / dictelem[l]['gwdt'][k]**2)
                
                setattr(gmodstat, 'sbrtextsbgrd', sbrtextsbgrd)
            sbrt['extsbgrd'] = []
            sbrt['extsbgrd'] = sbrtextsbgrd
            #objttimeprof.stopchro(gdat, gdatmodi, 'elemsbrtextsbgrd')
            
            if gdat.booldiag:
                cntppntschec = retr_cntp(gdat, sbrt['extsbgrd'])
                if np.amin(cntppntschec) < -0.1:
                    raise Exception('Point source spectral surface brightness is not positive-definite.')
        
    
    ## lensed surface brightness
    if boollens:
        
        #objttimeprof.initchro(gdat, gdatmodi, 'sbrtlens')
        
        if typeverb > 1:
            print('Evaluating lensed surface brightness...')
        
        if strgstat == 'this' or numbpopl > 0 and boolelemsbrtextsbgrdanyy:
            sbrt['bgrd'] = []
        if numbpopl > 0 and boolelemsbrtextsbgrdanyy:
            sbrt['bgrdgalx'] = []
        
        if dictchalinpt['numbener'] > 1:
            specsour = retr_spec(gdat, np.array([fluxsour]), sind=np.array([sindsour]))
        else:
            specsour = np.array([fluxsour])
        
        if numbpopl > 0 and boolelemsbrtextsbgrdanyy:
        
            if typeverb > 1:
                print('Interpolating the background emission...')

            sbrt['bgrdgalx'] = retr_sbrtsers(gdat, gdat.xposgrid[indxpixlelem[0]], gdat.yposgrid[indxpixlelem[0]], \
                                                                            xpossour, ypossour, specsour, sizesour, ellpsour, anglsour)
            
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
            if typeverb > 1:
                print('Not interpolating the background emission...')
            
            sbrt['lens'] = retr_sbrtsers(gdat, gdat.xposgrid - defl[gdat.indxpixl, 0], \
                                                   gdat.yposgrid - defl[gdat.indxpixl, 1], \
                                                   xpossour, ypossour, specsour, sizesour, ellpsour, anglsour)
            
            sbrt['bgrd'] = retr_sbrtsers(gdat, gdat.xposgrid, \
                                                   gdat.yposgrid, \
                                                   xpossour, ypossour, specsour, sizesour, ellpsour, anglsour)
            
        setattr(gmodstat, 'sbrtlens', sbrt['lens'])

        if gdat.booldiag:
            if not np.isfinite(sbrt['lens']).all():
                raise Exception('Lensed emission is not finite.')
            if (sbrt['lens'] == 0).all():
                raise Exception('Lensed emission is zero everywhere.')

        #objttimeprof.stopchro(gdat, gdatmodi, 'sbrtlens')
        
    ## host galaxy
    if typeemishost != 'none':
        #objttimeprof.initchro(gdat, gdatmodi, 'sbrthost')

        for e in indxsersfgrd:
            if typeverb > 1:
                print('Evaluating the host galaxy surface brightness...')
            
            if dictchalinpt['numbener'] > 1:
                spechost = retr_spec(gdat, np.array([fluxhost[e]]), sind=np.array([sindhost[e]]))
            else:
                spechost = np.array([fluxhost[e]])
            
            sbrt['hostisf%d' % e] = retr_sbrtsers(gdat, gdat.xposgrid, gdat.yposgrid, xposhost[e], \
                                                         yposhost[e], spechost, sizehost[e], ellphost[e], anglhost[e], serihost[e])
            
            setattr(gmodstat, 'sbrthostisf%d' % e, sbrt['hostisf%d' % e])
                
        #objttimeprof.stopchro(gdat, gdatmodi, 'sbrthost')
    
    ## total model
    #objttimeprof.initchro(gdat, gdatmodi, 'sbrtmodl')
    if typeverb > 1:
        print('Summing up the model emission...')
    
    sbrt['modlraww'] = np.zeros((dictchalinpt['numbener'], gdat.numbpixlcart, gdat.numbdqlt))
    for name in listnamediff:
        if name.startswith('back'):
            indxbacktemp = int(name[4:8])
            
            if gdat.typepixl == 'heal' and (typeevalpsfn == 'full' or typeevalpsfn == 'conv') and not boolunifback[indxbacktemp]:
                sbrttemp = getattr(gmod, 'sbrtbackhealfull')[indxbacktemp]
            else:
                sbrttemp = sbrtbacknorm[indxbacktemp]
           
            if boolspecback[indxbacktemp]:
                sbrt[name] = sbrttemp * bacp[indxbacpback[indxbacktemp]]
            else:
                sbrt[name] = sbrttemp * bacp[indxbacpback[indxbacktemp][gdat.indxener]][:, None, None]
        
        sbrt['modlraww'] += sbrt[name]
        
        if gdat.booldiag:
            if np.amax(sbrttemp) == 0.:
                raise Exception('')

    # convolve the model with the PSF
    if convdiffanyy and (typeevalpsfn == 'full' or typeevalpsfn == 'conv'):
        sbrt['modlconv'] = []
        # temp -- isotropic background proposals are unnecessarily entering this clause
        if typeverb > 1:
            print('Convolving the model image with the PSF...') 
        sbrt['modlconv'] = np.zeros((dictchalinpt['numbener'], gdat.numbpixl, gdat.numbdqlt))
        for ii, i in enumerate(gdat.indxener):
            for mm, m in enumerate(gdat.indxdqlt):
                if gdat.strgcnfg == 'pcat_ferm_igal_simu_test':
                    print('Convolving ii, i, mm, m')
                    print(ii, i, mm, m)
                if gdat.typepixl == 'cart':
                    if gdat.numbpixl == gdat.numbpixlcart:
                        sbrt['modlconv'][ii, :, mm] = convolve_fft(sbrt['modlraww'][ii, :, mm].reshape((gdat.numbsidecart, gdat.numbsidecart)), \
                                                                                                                             objtpsfnconv[mm][ii]).flatten()
                    else:
                        sbrtfull = np.zeros(gdat.numbpixlcart)
                        sbrtfull[gdat.indxpixlrofi] = sbrt['modlraww'][ii, :, mm]
                        sbrtfull = sbrtfull.reshape((gdat.numbsidecart, gdat.numbsidecart))
                        sbrt['modlconv'][ii, :, mm] = convolve_fft(sbrtfull, objtpsfnconv[mm][ii]).flatten()[gdat.indxpixlrofi]
                    indx = np.where(sbrt['modlconv'][ii, :, mm] < 1e-50)
                    sbrt['modlconv'][ii, indx, mm] = 1e-50
                if gdat.typepixl == 'heal':
                    sbrt['modlconv'][ii, :, mm] = hp.smoothing(sbrt['modlraww'][ii, :, mm], fwhm=fwhm[i, m])[gdat.indxpixlrofi]
                    sbrt['modlconv'][ii, :, mm][np.where(sbrt['modlraww'][ii, :, mm] <= 1e-50)] = 1e-50
        
        setattr(gmodstat, 'sbrtmodlconv', sbrt['modlconv'])
        # temp -- this could be made faster -- need the copy() statement because sbrtdfnc gets added to sbrtmodl afterwards
        sbrt['modl'] = np.copy(sbrt['modlconv'])
    else:
        if typeverb > 1:
            print('Skipping PSF convolution of the model...')
        sbrt['modl'] = np.copy(sbrt['modlraww'])
    
    if typeverb > 1:
        print('sbrt[modl]')
        summgene(sbrt['modl'])

    ## add PSF-convolved delta functions to the model
    if numbpopl > 0 and boolelemsbrtdfncanyy:
        sbrt['modl'] += sbrt['dfnc']
    #bjttimeprof.stopchro(gdat, gdatmodi, 'sbrtmodl')
    
    if typeverb > 1:
        print('sbrt[modl]')
        summgene(sbrt['modl'])
    
    #dictchaloutp['objttimeprof'] = objttimeprof

    return dictchaloutp


