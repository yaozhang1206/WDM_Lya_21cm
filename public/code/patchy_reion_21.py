import numpy as np
from scipy import interpolate
from scipy import integrate
import pickle


class P_21_obs:

    """
        I am also pickling this guy!
    """
    
    def __init__(self, params):

        self.h = params['h']
        self.Obh2 = params['Obh2']
        self.Och2 = params['Och2']
        self.mnu2 = params['mnu']
        self.ns = params['ns']
        self.alpha_s = params['alphas']
        self.tau_re = params['taure']
        self.bHI = params['bHI'] # fiducial is 2.82 (range 3.5<z<4.5)
        self.OHI = params['OHI'] / 1.e3 # 1.18 fiducial //
        self.OMh2 = self.Obh2 + self.Och2
        self.fast_realization = params['fast-realization']
        self.fast_model = params['fast-model']
        self.gadget_realization = params['gadget-realization']
        self.gadget_model = params['gadget-model']
        # hyper parameters
        self.zres = np.arange(6., 12.+0.1, 0.01)
        file = open('../pickles/pat_'+self.fast_realization+'_'+self.fast_model+'.pkl', 'rb')
        self.PmxH = pickle.load(file)
        file.close()


    # adding the bias and omega as functions here
    def OHI_func(self, z):
        if 3.49 < z < 4.5:
            return 1.18e-3
        elif 4.5 <= z < 5.51:
            return 0.98e-3
            
    def bHI_func(self, z):
        if 3.49 < z < 4.5:
            return 2.82
        elif 4.5 <= z < 5.51:
            return 3.18

    def Tb_mean(self, z):
        '''
        output in units of mK
        '''
        
        return 27*np.sqrt((1+z)/10*0.15/self.OMh2)*(self.OHI_func(z)*self.h**2/0.023)
        

    def f(self, z):
    # Growth rate = OM(z)^0.545 = (OM*(1+z)^3/(OM*(1+z)^3+(1-OM)))^0.545 ref: 1709.07893
        OM = (self.Obh2+self.Och2)/self.h**2
        return (OM*(1+z)**3/(OM*(1+z)**3+1-OM))**0.545

    def beta_21(self, z):
        # returns the rsd parameter
        return self.f(z) / self.bHI_func(z)


    def reion_his(self,filename):
        data = np.loadtxt(filename)
        z_rh =data.T[0]
        xH = data.T[1]
        return interpolate.interp1d(z_rh,xH)
        

    def rho_HI(self):
        #rho_HI_func_{Gadge_realization}_{Gadget model}.pkl
        with open('../pickles/rho_HI_func_'+self.gadget_realization+'_'+self.gadget_model+'.pkl','rb') as f:
           rho_HI = pickle.load(f)
           
        return rho_HI

    def reion_mid(self):
        xi_arr = self.reion_his('../data/21cmFAST/xH_21cm_ave_'+self.fast_model+'.txt')
        for z in np.arange(12.0,6.0,-0.01):
            if (xi_arr(z)<0.5 and xi_arr(z+0.01)>0.5): break
            
        z = z + ((0.5-(xi_arr(z)))/(xi_arr(z+0.01)-xi_arr(z)))*0.01
        
        return z

    def dpsi_dz(self, z):
        z_re_mean = self.reion_mid()
        rho_HI = self.rho_HI()
        dpsi_dz = np.gradient(np.log(rho_HI(self.zres,z)/rho_HI(z_re_mean,z)),self.zres)
        dpsi_dz[np.isnan(dpsi_dz)] = 0
        return dpsi_dz


    def Xi_for_plot(self, z_re, z_obs):
        z_re_mean = self.reion_mid()
        rho_HI = self.rho_HI()
        Xi = np.log(rho_HI(z_re, z_obs) / rho_HI(z_re_mean, z_obs))
        return Xi

    def P_m_Xi(self, z, k):
        dpsi_dz = self.dpsi_dz(z)
        P_m_Xi = -integrate.simps(dpsi_dz * self.PmxH(self.zres, k) * (np.ones(self.zres.shape) + self.zres) / (1. + z), self.zres) # D prop to a
        
        return P_m_Xi
        
        
    def P_reion(self, z, k, mu):
        # normalized, i.e. no mK^2
        P_patchy = 2 * (self.bHI_func(z) + mu**2 * self.f(z)) * self.P_m_Xi(z, k)

        return P_patchy
        
    def pickle_this(self):
        # hhopefully it is faster to get the function than to compute it
        file = open('../pickles/p_mXi_'+self.fast_realization+'_'+self.fast_model+'_'+self.gadget_realization+'_'+self.gadget_model+'.pkl', 'wb')
        pickle.dump(self.P_m_Xi, file)
        file.close()
        print('PickleD!')
        return 
