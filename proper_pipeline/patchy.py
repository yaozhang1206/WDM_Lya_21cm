import numpy as np
from scipy import interpolate
import pickle

"""
    Build the matter cross xHI, we want to pickle this extrapolated function to save time!
    
    Only needs to be run once per model!
    
"""


class P_bubble(object):
    
    def __init__(self, params):
        # lya and 21cm needs this, so it will be good to separate it and build it once
        model = params['fast-model'] # know which file is getting called from 21cmFAST
        realization = params['fast-realization'] # know which realization, or if it is the average one
        # for average it should be avg
        # model just covers reionization histories but it could be more than that.
        filename = './data/21cmfast/matter_cross_HI_'+realization+'_'+model+'.txt'
        # grab data
        Pc = np.loadtxt(filename)
        # transform from dimensionless to Mpc^3
        Pk = Pc.T[2] * 2 * np.pi**2 / Pc.T[1]**3
        # interpolate power
        self.P_mxHI = interpolate.LinearNDInterpolator(Pc[:,0:2], np.array(Pk).flatten())
        # file for our pickle
        file = open('./pickle/pat_'+realization+'_'+model+'.pkl', 'wb')
        # dump information to file
        pickle.dump(self.P_mxHI, file)
        file.close()
        
        
