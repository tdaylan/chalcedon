import sys

import numpy as np

import chalcedon


def cnfg_microlens():
    
    dictchalinpt = dict()
    dictchalinpt['sherextr'] = 0.3
    dictchalinpt['sangextr'] = 0.3
    
    dictchalinpt['xpossubh'] = np.array([1.])
    dictchalinpt['ypossubh'] = np.array([1.])
    dictchalinpt['defssubh'] = np.array([1.])
    dictchalinpt['ascasubh'] = np.array([1.])
    dictchalinpt['acutsubh'] = np.array([1.])

    dictchaloutp = chalcedon.retr_caustics( \
                                           dictchalinpt=dictchalinpt, \
                                          )
    

globals().get(sys.argv[1])(*sys.argv[2:])

