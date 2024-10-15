import numpy as np
import patchy_reion_21 as hey
import puma as pu
import skalow as sk
import pickle
import Pm_DM as pm

"""
    Now we are using pickle for saving memories
"""

class theory_P_21(object):

    def __init__(self,params):
        telescope = params['telescope']
        t_int = params['t_int']
        beam = params['beam'] # used to calculate volume
        h = params['h']
        # we want WDM in the dictionary, so we pass it along to people that needed.
        Omega_r = 8.6e-5
        Omega_m = params['Och2'] / h**2
        # let's get to the unpickling
        fast_model = params['fast-model'] # this can be things like 'early' or '3keV_s8' or 'cdm_s8'
        # in the future it could be 'early_3keV'
        fast_realization = params['fast-realization'] # e.g. 'r1', or 'ave'
        gadget_realization = params['gadget-realization']
        gadget_model = params['gadget-model']
        f21 = open('../pickles/p_mXi_'+fast_realization+'_'+fast_model+'_'+gadget_realization+'_'+gadget_model+'.pkl', 'rb')
        self.P_m_Xi = pickle.load(f21)
        f21.close()
        # get lya related stuff
        self.sigma8 = params['sigma8'] # note that we also use As, fid = 0.8159, need to be careful with that!
        # we grab the wdm value in case someone forgot which model is this
        self.m_wdm = params['m_wdm']
        if telescope == 'puma':
            self.tel = pu.puma(t_int, beam, Omega_m, h, Omega_r)
        elif telescope == 'skalow':
            self.tel = sk.skalow(t_int, beam, Omega_m, h, Omega_r)
        # and for the HI
        # we need an instance to get the bias/rsd
        self.P_21_hey = hey.P_21_obs(params)
        self.cosmo = pm.P_matter(params)
        

    # Capability to compute the 21 auto
    def HIHI_base_Mpc_norm(self, z, k_Mpc, mu):
        """ 3D base 21 cm power spectrum """
        b_21 = self.P_21_hey.bHI_func(z)
        beta_21 = self.P_21_hey.beta_21(z)
        P_m = self.cosmo.P_m_Mpc(k_Mpc, z)
        return b_21**2 * (1. + beta_21 * mu**2)**2 * P_m
        
    def HIHI_reio_Mpc_norm(self, z, k_Mpc, mu):
        """ reio term in 21cm auto-power spectrum """
        b_21 = self.P_21_hey.bHI_func(z)
        beta_21 = self.P_21_hey.beta_21(z)
        return 2. * b_21 * (1. + beta_21 * mu**2) * self.P_m_Xi(z, k_Mpc)

    def P3D_21_Mpc_norm(self, z, k_Mpc, mu):
        # returns the signal
        """ Turn reionization on or off """
        return self.HIHI_base_Mpc_norm(z,k_Mpc,mu) + self.HIHI_reio_Mpc_norm(z,k_Mpc,mu)
        
    def Total_P_21_Mpc_norm(self, z, k_Mpc, mu):
        """ Returns the total power spectrum including noise and memory of reionization """
        P3D = self.P3D_21_Mpc_norm(z, k_Mpc, mu)
        PN = self.tel.noise_power_Mpc(z, k_Mpc, mu) # normalize
        PN = PN / self.P_21_hey.Tb_mean(z)**2
        P_tot = P3D + PN
        return P_tot


    def Var_autoHI_Mpc(self, z, k_Mpc, mu, epsilon):
        """ Computes the variance for the 21cm auto """
        dmu = 0.2
        sigma_IM = self.Total_P_21_Mpc_norm(z, k_Mpc, mu)
        # volume
        V_Mpc = self.tel.survey_vol_Mpc(z)
        # number of modes
        Nmodes = V_Mpc * k_Mpc * k_Mpc * k_Mpc * epsilon * dmu / (2. * np.pi * np.pi)
        var21 = 2. * sigma_IM**2 / Nmodes
        return var21
    
    def Var_autoHI_Mpc_yao(self, z, k_Mpc, mu, dz, dk, dmu):
        """ Computes the variance for the 21cm auto """
        sigma_IM = self.Total_P_21_Mpc_norm(z, k_Mpc, mu)
        # volume
        V_Mpc = self.tel.survey_vol_Mpc_eqn(z, dz)
        # number of modes
        Nmodes = V_Mpc * k_Mpc * k_Mpc * dk * dmu / (2. * np.pi * np.pi)
        var21 = 2. * sigma_IM**2 / Nmodes
        return var21

