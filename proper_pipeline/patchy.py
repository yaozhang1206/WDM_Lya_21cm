import numpy as np
from scipy import interpolate
import pickle
import Pm_DM as pm

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
        filename = '../data/21cmfast/cross_21cm_'+realization+'_'+model+'.txt'
        # grab data
        Pc = np.loadtxt(filename)
        # transform from dimensionless to Mpc^3
        Pk = Pc.T[2] * 2 * np.pi**2 / Pc.T[1]**3
        # interpolate power
        self.P_mxHI = interpolate.LinearNDInterpolator(Pc[:,0:2], np.array(Pk).flatten())
        # before we pickle we need to deal with biasing procedure
        self.k_plot = np.logspace(-4,1,1000)
        self.k_min = Pc.T[1][3]
        self.k_max = Pc.T[1][-1]
        self.cosmo = pm.P_matter(params)
        
        
    def P_mxHI_func(self, z, k_Mpc):
        """ implementing biasing procedure that it is required for 21cm """
        if k_Mpc <= self.k_max and k_Mpc >= self.k_min:
            return self.P_mxHI(z, k_Mpc)
        elif k_Mpc > self.k_max:
            return 0
        elif k_Mpc < self.k_min:
            bias = self.P_mxHI(z, self.k_min) / (self.cosmo.P_m_Mpc(self.k_min, z))
            return bias * self.cosmo.P_m_Mpc(k_Mpc, z)
        
        
    def pickle_this(self):
        # file for our pickle
        file = open('../pickle/pat_'+realization+'_'+model+'.pkl', 'wb')
        # dump information to file
        pickle.dump(self.P_mxHI_func, file)
        file.close()
        return print('PickleD!')
