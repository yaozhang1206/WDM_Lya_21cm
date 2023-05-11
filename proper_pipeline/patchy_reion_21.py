import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg
import sys
from math import floor
import pickle
import Pm_DM as pm


# OM = 0.3089
H0 = 67.74 # km/s/Mpc
# h = 4.135667696e-15 # Planck constant in units of eV/Hz
# nu_0 = 13.6/h
# c = 3.0e10 # light speed in units of cm
G = 6.67e-11 # kg^-1 m^3 s^-2
# Yp = 0.249 # Helium abundance
# mH = 1.6729895e-24 # g
# bolz_k = 8.617333262145e-5 # Bolzmann constant in units of eV/K
rho_crit=3*H0**2/8/np.pi/G*H0/100 # M_solar/(Mpc)^3/h

# rhom=rho_crit*OM

# parsec=3.085677581e16 # m per parsec
# H_0=67.74 # Hubble constants now, 67.74 km/s/mpc
# G=4.30091e-9  #6.674×10−11 m3*kg−1*s−2 ### 4.30091(25)×10−3 Mpc*M_solar-1*(km/s)^2
# solar_m= 1.98847e30 #(1.98847±0.00007)×10^30 kg

# z=5.5

# Omega_m=0.3089 # Omega_m = 0.3089+-0.0062
# rhom=rho_crit*Omega_m #*(1+z)**3
# f_b=0.17 # baryon fraction


class P_21_obs:

    """
        I am also pickling this guy!
    """
    
    def __init__(self, params):

        self.h = params['h']
        # self.h=0.6774
        self.Obh2 = params['Obh2']
        self.Och2 = params['Och2']
        self.mnu2 = params['mnu']
        self.As = params['As'] / 1.e9 # As read in 10^9 * As
        self.ns = params['ns']
        self.alpha_s = params['alphas']
        self.tau_re = params['taure']
#         self.OHIh2s = params['O_HIs']*0.6774**2
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

        # rho_HI_func_{Gadge_realization}_{Gadget model}.pkl
        # unfortunately can't change yet until Yao starts generating pickles, but I
        # should make a pickle farmer
        with open('../pickles/rho_HI_func_'+self.gadget_realization+'_'+self.gadget_model+'.pkl','rb') as f:
           rho_HI = pickle.load(f, encoding='latin1')
           
        return rho_HI

    def reion_mid(self):
        xi_arr = self.reion_his('../data/21cmFAST/xH_21cm_'+self.fast_realization+'_'+self.fast_model+'.txt')
        for z in np.arange(9.0,6.0,-0.01):
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
        return print('PickleD!')
