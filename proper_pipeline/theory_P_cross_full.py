import numpy as np
import patchy_reion_21 as hey
import observed_3D as obs
import puma as pu
import skalow_v2 as sk
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
        fast_model = params['fast-model'] # this can be things like 'early' or '3keV' or 'cdm'
        # in the future it could be 'early_3keV'
        fast_realization = params['fast-realization'] # e.g. 'r1', or 'avg'
        gadget_model = params['gadget-model'] # e.g. '3keV', 'cdm'
        gadget_realization = params['gadget-realization'] # e.g. 'r2' or 'avg'
        flya = open('./pickle/p_mpsi_'+fast_realization+'_'+fast_model+'_'+gadget_realization+'_'+gadget_model+'.pkl', 'rb')
        self.P_m_psi = pickle.load(flya)
        flya.close()
        f21 = open('./pickle/p_mXi_'+fast_realization+'_'+fast_model+'.pkl', 'rb')
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
        self.k_IM_Mpc = 0.01
        self.Nk = 100
        self.Nmu = 5
        self.kmax_hey = 0.4
        self.dz = 0.2
        
        
    # start with theoretical signal
    def Lya_HI_base_Mpc_norm(self, z, k_Mpc, mu):
        """ Computes the base (no reio term) cross-correlation term """
        b_F = self.Forest.my_P.flux_bias(z, self.Forest.my_P.our_sigma8)
        b_21 = self.P_21_hey.bHI
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
        b_21 = self.P_21_hey.bHI
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
        
    # Capability to compute the 21 auto
    def HIHI_base_Mpc_norm(self, z, k_Mpc, mu):
        """ 3D base 21 cm power spectrum """
        b_21 = self.P_21_hey.bHI
        beta_21 = self.P_21_hey.beta_21(z)
        P_m = self.Forest.my_P.cosmo.P_m_Mpc(k_Mpc, z)
        return b_21**2 * (1. + beta_21 * mu**2)**2 * P_m
        
    def HIHI_reio_Mpc_norm(self, z, k_Mpc, mu):
        """ reio term in 21cm auto-power spectrum """
        b_21 = self.P_21_hey.bHI
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
        PN = PN / self.P_21_hey.Tb_mean(z)
        P_tot = P3D + PN
        return P_tot
 
    def Var_cross_Mpc(self, z, k_Mpc, mu, epsilon, Pw2D=None, PN_eff=None, PmemHI=None, PmemF=None):
        """ Computes the variance for the cross correlation """
        # I changed dk_Mpc for epsilon = dk_Mpc / k_Mpc so it is now log binning
        # again for fixed mu
        dmu = 0.2
        kp_Mpc = k_Mpc * mu
        kt_Mpc = k_Mpc * np.sqrt(1. - mu**2)
        # transform
        kp_kms = kp_Mpc / self.dkms_dMpc
        kt_deg = kt_Mpc * self.dMpc_ddeg
        sigma_Forest = self.Forest.TotalFluxP3D_Mpc(kt_deg, kp_kms, Pw2D, PN_eff)
        sigma_IM = self.Total_P_21_Mpc_norm(z, k_Mpc, mu)
        sigma_tot = 0.5 * (self.Total_P_cross_Mpc_norm(z, k_Mpc, mu, PmemHI, PmemF)**2 + sigma_IM * sigma_Forest)
        # need to add the number of modes too.
        # so for volume let's use the smaller volume, the 21cm telescope
        V_Mpc = self.tel.survey_vol_Mpc(z)
        # and the number of modes in the volume is
#        Nmodes = V_Mpc * k_Mpc * k_Mpc * dk_Mpc * dmu / (2. * np.pi * np.pi) # original
        Nmodes = V_Mpc * k_Mpc * k_Mpc * k_Mpc * epsilon * dmu / (2. * np.pi * np.pi)
        varP = 2.0 * sigma_tot / Nmodes
        return varP

    def Var_autoHI_Mpc(self, z, k_Mpc, mu, epsilon):
        """ Computes the variance for the 21cm auto """
        dmu = 0.2
        sigma_IM = self.Total_P_21_Mpc_norm(z, k_Mpc, mu)
        # volume
        V_Mpc = self.tel.survey_vol_Mpc(z)
        # number of modes
        Nmodes = V_Mpc * k_Mpc * k_Mpc * epsilon * dmu / (2. * np.pi * np.pi)
        var21 = 2. * sigma_IM**2 / Nmodes
        return var21

    # let's start the capability to compute the spherically-average one
    # will do every power at the same time
    def sph_average_p(self, z, k_Mpc, epsilon, Pw2D=None, PN_eff=None):
        dmu = 0.2
        mu = [0.1, 0.3, 0.5, 0.7, 0.9]
        # to speed up things
        if Pw2D==None:
            np_eff, Pw2D, PN_eff = self.Forest.EffectiveDensityAndNoise()
            print('Had to do it myself')
        result = np.zeros(13)
        # the order is Base_x, reio_x^HI, reio_x^F, tot_x, 1/var_x, Base_21, reio_21, tot_21
        for i in range(0,len(mu)):
            result[0] += dmu * self.Lya_HI_base_Mpc_norm(z, k_Mpc, mu[i])
            # storing for speed lol
            temp_memHI = self.cross_HI_memory_Mpc_norm(z, k_Mpc, mu[i])
            temp_memF = self.cross_F_memory_Mpc_norm(z, k_Mpc, mu[i])
            result[1] += dmu * temp_memHI
            result[2] += dmu * temp_memF
#            print('Did the cross part')
            # not integrating total because I can add
            # note that variance includes the dmu in the number of modes, so I can just do
            result[4] += 1. / self.Var_cross_Mpc(z, k_Mpc, mu[i], epsilon, Pw2D=Pw2D, PN_eff=PN_eff, PmemHI=temp_memHI, PmemF=temp_memF)
#            print('Did the variance for the cross')
            result[5] += dmu * self.HIHI_base_Mpc_norm(z, k_Mpc, mu[i])
            result[6] += dmu * self.HIHI_reio_Mpc_norm(z, k_Mpc, mu[i])
            # and the 21 cm variance is
            result[8] += 1. / self.Var_autoHI_Mpc(z, k_Mpc, mu[i], epsilon)
#            print('Did the 21 cm auto parts')
            # now for lya
            result[9] += dmu * self.Forest.my_P.LyaLya_base_Mpc_norm(z, k_Mpc, mu[i])
            result[10] += dmu * self.Forest.my_P.LyaLya_reio_Mpc_norm(z, k_Mpc, mu[i])
            # I want to use the Lya too, but we need the variance of both too...
            result[12] += 1. / self.Forest.VarFluxP3D_Mpc(k_Mpc, mu[i], epsilon, dmu, Pw2D=Pw2D, PN_eff=PN_eff)
        # get the totals
        result[3] = result[0] + result[1] + result[2]
        result[7] = result[5] + result[6]
        result[11] = result[9] + result[10]
        return result
        
