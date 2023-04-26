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
        self.model = params['fast-model'] # know which file is getting called from 21cmFAST
        self.realization = params['fast-realization'] # know which realization, or if it is the average one
        # for average it should be avg
        # model just covers reionization histories but it could be more than that.
        filename = '../data/21cmfast/cross_21cm_'+self.realization+'_'+self.model+'.txt'
        # grab data
        Pc = np.loadtxt(filename)
        # transform from dimensionless to Mpc^3
        Pk = Pc.T[2] * 2 * np.pi**2 / Pc.T[1]**3
        # interpolate power but later
        self.P_mxHI = interpolate.LinearNDInterpolator(Pc[:,0:2], np.array(Pk).flatten())
        # before we pickle we need to deal with biasing procedure
        self.k_plot = np.logspace(-3,0,100)
        self.k_min = Pc.T[1][3]
        self.k_max = Pc.T[1][-1]
        self.cosmo = pm.P_matter(params)
        # situation is a little more problematic, due to not being able to pickle classy...
        # here we go
        self.k_temp = np.array(Pc.T[1][:]).flatten()
        self.z_temp = np.array(Pc.T[0][:]).flatten()
        self.z_temp = np.unique(self.z_temp)
        self.P_temp = np.array(Pc.T[2][:]).flatten()
#        print(self.z_temp)
#        print(self.k_temp)
        self.temp_table = np.zeros((len(self.z_temp)*len(self.k_temp),3))
        print('This patchy pickle will take some time just because I did not try to optimize this procedure... :P')
        for i in range(0, len(self.z_temp)):
            for j in range(0, len(self.k_plot)):
                self.temp_table[i*len(self.k_plot) + j, 0] = self.z_temp[i]
                self.temp_table[i*len(self.k_plot) + j, 1] = self.k_plot[j]
                self.temp_table[i*len(self.k_plot) + j, 2] = self.P_mxHI_vegetable(self.z_temp[i], self.k_plot[j])
        np.savetxt('../data/21cmfast/matter_cross_21cm_'+self.realization+'_'+self.model+'.txt', self.temp_table, fmt='%e', delimiter=' ')
        
        
        
        
    def P_mxHI_vegetable(self, z, k_Mpc):
        """ implementing biasing procedure that it is required for 21cm """
        if k_Mpc <= self.k_max and k_Mpc >= self.k_min:
            return self.P_mxHI(z, k_Mpc)
        elif k_Mpc > self.k_max:
            return 0
        elif k_Mpc < self.k_min:
            bias = self.P_mxHI(z, self.k_min) / (self.cosmo.P_m_Mpc(self.k_min, z))
            return bias * self.cosmo.P_m_Mpc(k_Mpc, z)
        
       
       
    def pickle_this(self):
        temp = np.loadtxt('../data/21cmfast/matter_cross_21cm_'+self.realization+'_'+self.model+'.txt')
        Pk = temp.T[2]
        self._P_mxHI = interpolate.LinearNDInterpolator(temp[:,0:2], np.array(Pk).flatten())
        # file for our pickle
        file = open('../pickles/pat_'+self.realization+'_'+self.model+'.pkl', 'wb')
        # dump information to file
        pickle.dump(self._P_mxHI, file)
        file.close()
        return print('PickleD!')
