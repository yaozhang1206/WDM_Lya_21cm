import numpy as np
import patchy_reion_21 as hey
import observed_3D as obs
import puma as pu
import skalow_v2_fov as sk
import pickle
import Pm_DM as pm

"""
    For SNR for cross-correlation of 21cm and lya. This is the full version it will need to call a class from Heyang for every z,k,mu... or just put a random one and update!
    
    Now we are using pickle for our memories
"""

class theory_P_cross(object):
    # cosmology may not be exactly the same, this may be problematic for this simple comparison. However, cosmology should be very similar.
    
    # noise for Lya is going to be a problem due to redshift binning

    def __init__(self,params):
        telescope = params['telescope']
        t_int = params['t_int']
        beam = params['beam'] # used to calculate volume?
        h = params['h']
        # we want WDM in the dictionary, so we pass it along to people that needed.
        Omega_r = 8.6e-5
        Omega_m = params['Och2'] / h**2
        # let's get to the unpickling
        fast_model = params['fast-model'] # this can be things like 'early' or '3keV_s8' or 'cdm_s8'
        # in the future it could be 'early_3keV'
        fast_realization = params['fast-realization'] # e.g. 'r1', or 'ave'
        gadget_model = params['gadget-model'] # e.g. '3keV_s8', 'cdm_s8'
        gadget_realization = params['gadget-realization'] # e.g. 'r2' or 'ave'
        flya = open('../pickles/p_mpsi_'+fast_realization+'_'+fast_model+'_'+gadget_realization+'_'+gadget_model+'.pkl', 'rb')
        self.P_m_psi = pickle.load(flya)
        flya.close()
        f21 = open('../pickles/p_mXi_'+fast_realization+'_'+fast_model+'_'+gadget_realization+'_'+gadget_model+'.pkl', 'rb')
        self.P_m_Xi = pickle.load(f21)
        f21.close()
        # get lya related stuff
        self.sigma8 = params['sigma8'] # note that we also use As, fid = 0.8159, need to be careful with that!
        # we grab the wdm value in case someone forgot which model is this
        self.m_wdm = params['m_wdm']
        # I will  need noise for flux too
        self.Forest = obs.observed_3D(params)
        # let's change redshift too but make this later, first no noise
        self.Forest.lmin = 3501 + 200. * 10
        self.Forest.lmax = 3701 + 200. * 10
        # so it starts with a z_mean of 3.61
        """ to change redshift bin please change the lrange of the forest """
        if telescope == 'puma':
            self.tel = pu.puma(t_int, beam, Omega_m, h, Omega_r)
        elif telescope == 'skalow':
            self.tel = sk.skalow(t_int, beam, Omega_m, h, Omega_r)
        self.z = self.Forest.mean_z()
        print('Currently at ', self.Forest.mean_z())
        self.dkms_dMpc = self.Forest.convert.dkms_dMpc(self.Forest.mean_z())
        self.dMpc_ddeg = self.Forest.convert.dMpc_ddeg(self.Forest.mean_z())
        # and for the HI
        # we need an instance to get the bias/rsd
        self.P_21_hey = hey.P_21_obs(params)
        self.cosmo = pm.P_matter(params)
        
        
    # start with theoretical signal
    def Lya_HI_base_Mpc_norm(self, z, k_Mpc, mu):
        """ Computes the base (no reio term) cross-correlation term """
        b_F = self.Forest.my_P.flux_bias(z, self.Forest.my_P.our_sigma8)
        b_21 = self.P_21_hey.bHI_func(z)
        beta_F = self.Forest.my_P.beta_rsd(z, self.Forest.my_P.our_sigma8)
        beta_21 = self.P_21_hey.beta_21(z)
        P_m = self.Forest.my_P.cosmo.P_m_Mpc(k_Mpc, z)
        verbose = 0
        if verbose == 1:
            print('Flux bias: ', b_F)
            print('Beta F: ', beta_F)
            print('b_21: ', b_21)
            print('beta_21: ', beta_21)
            print('Matter power: ', P_m)
        return b_F * b_21 * (1. + beta_F * mu**2) * (1. + beta_21 * mu**2) * P_m
        
        
  
    def cross_HI_memory_Mpc_norm(self, z, k_Mpc, mu):
        """ Computes the memory of reionization term sourced by dense regions """
        b_F = self.Forest.my_P.flux_bias(z, self.Forest.my_P.our_sigma8)
        beta_F = self.Forest.my_P.beta_rsd(z, self.Forest.my_P.our_sigma8)
        verbose = 0
        if verbose == 1:
            print('Flux bias: ', b_F)
            print('Beta F: ', beta_F)
            print('P_m_Xi: ', self.P_m_Xi)
        return b_F * (1. + beta_F * mu**2) * self.P_m_Xi(z, k_Mpc)
        
    def cross_F_memory_Mpc_norm(self, z, k_Mpc, mu):
        """ Computes the memory of reionization term sourced by underdense regions """
        b_21 = self.P_21_hey.bHI_func(z)
        beta_21 = self.P_21_hey.beta_21(z)
        bias_G = self.Forest.my_P.b_gamma(z)
        return b_21 * (1. + beta_21 * mu**2) * bias_G * self.P_m_psi(z, k_Mpc)
 
    def Total_P_cross_Mpc_norm(self, z, k_Mpc, mu, PmemHI=None, PmemF=None):
        # returns total signal
        """ Turn reionization on or off """
        if PmemHI == None:
            return self.Lya_HI_base_Mpc_norm(z, k_Mpc, mu) + self.cross_HI_memory_Mpc_norm(z, k_Mpc, mu) + self.cross_F_memory_Mpc_norm(z, k_Mpc, mu)
        else:
            return self.Lya_HI_base_Mpc_norm(z, k_Mpc, mu) + PmemHI + PmemF
    
        

