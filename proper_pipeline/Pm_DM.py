import numpy as np
import matplotlib.pyplot as plt
from classy import Class
from scipy.interpolate import interp1d

"""
    Build the matter power spectrum using classy
"""


class P_matter(object):
    
    def __init__(self, params):
        # now time to build class model
        # Define your cosmology (what is not specified will be set to CLASS default parameters, see explanatory.ini for specific information regarding all parameters)
        # for reference these are the parameters that go into the fisher matrix
        
        """ need to be careful with Lyman alpha since we have a different parametrization (sigma8) """
        # getting wdm stuff
        self.m_WDM_keV = params['m_wdm']
        
        self.params = {
            'output': 'mPk',
            'T_cmb': 2.7255,
            'N_ur': 3.046,
            'N_ncdm': 1,
#            'ncdm_psd_parameters': (0.3, 0.5, 0.05),
            'deg_ncdm': 1,
            'Omega_k': 0,
            'YHe' : 'BBN',
            'recombination': 'RECFAST',
            'reio_parametrization': 'reio_camb',
            'reionization_exponent': 1.5,
            'reionization_width': 0.5,
            'helium_fullreio_redshift': 3.5,
            'helium_fullreio_width': 0.5,
            'gauge': 'synchronous',
            'P_k_ini type': 'analytic_Pk',
            'P_k_max_1/Mpc': 10,
            'format': 'CLASS',
            'z_max_pk': 5.5,
            # here comes the changing ones
            'h': params['h'],
            'n_s': params['ns'],
            'A_s': params['As'] / 1.e9,
            'm_ncdm': params['mnu'],
            'alpha_s': params['alphas'],
            'tau_reio': params['taure'],
            'Omega_b': params['Obh2'] / (params['h']**2),
            'Omega_cdm': params['Och2'] / (params['h']**2),
        } # values chosen consistent with planck cosmology, #P_k_max_1/Mpc was 5 before
        
        # Create an instance of the CLASS wrapper, this class we call for solving Boltzmann equations
        self.cosmo = Class()
        # Set the parameters to the cosmological code, i.e. tells the wrapper to pass this dictionary to the c code
        self.cosmo.set(self.params)
        # Run the whole code, i.e. computes P_m(k,z)
        self.cosmo.compute()
        # Let's make a function to get us our new computed P_m, but first let's get little h
        self.h = self.cosmo.h()
        print('Finished prepping the model for P_m')


    def alpha_WDM(self):
        """
        Returns the suppression scale due to WDM models free streaming more than CDM, in Mpc.

        Inputs (not really since they're class attributes : m_WDM [keV] (mass of the WDM candidate),
                g_WDM (degrees of freedom, i.e. 1.5) but for thermal relics it does nothing, so let's get rid off it
                Omega_DM (dark matter density of the Universe)
                
        Outputs: alpha [Mpc]
        """
        return 0.049 / self.params['h'] * pow(self.m_WDM_keV,-1.11) * pow(self.params['Omega_cdm'] / 0.25,0.11) * pow(self.params['h'] / 0.7,1.22)

    def T_WDM(self, k_Mpc):
        """
        Transfer function of WDM, taken from Viel et al. (~2006), remember that it is dimensionless

        Inputs: k [Mpc^-1], note that no redshift nor little h in the input

        Outputs: T_WDM
        """
        mu = 1.12
        alpha = self.alpha_WDM()
        pa = 1.0 + pow(alpha * k_Mpc, 2 * mu)
        return pow(pa, -5. / mu)

    def P_m_Mpc(self,k_Mpc,z):
        """
        Returns the 3D matter power spectrum obtained from CLASS in units of Mpc^3 (no little h!). Note that this is a function of redshift too.
        
        Inputs: k [h Mpc^-1], z
        
        Outputs: P_m_CDM [Mpc^3]
        """
        # so actually pk needs k_Mpc as input and throws P_Mpc, so [Mpc^3] units, no h.
        # note that in case of cdm m_wdm -> infinity, thus transfer function -> 1
        return self.cosmo.pk_lin(k_Mpc, z) * self.T_WDM(k_Mpc) * self.T_WDM(k_Mpc)

    def sigma(self, M_h, z):
        """
        Returns sigma at z needed for our HMF
        
        Inputs: M_h the halo mass and redshift
        
        Outputs: sigma
        """
#        R = 8. / self.h
        # critical density
        rho_crit = 1.879e-29 # g cm^{-3} h^2
        rho_crit = rho_crit * self.h * self.h
        # convert to solar masses
        rho_crit = rho_crit / 2.e33
        # and to Mpc
        dcm_dMpc = 3.0857e24
        rho_crit = rho_crit * dcm_dMpc**3
        Omega_m = 0.2602 + 0.0486
        rho_mean_z = Omega_m * rho_crit * (1. + z**3)
        M_h = M_h
#        print('Halo mass: in 10^8 solar masses ', M_h / 1.e8)
#        print('mean density: ', rho_mean_z)
        R = 3. * M_h / (4. * np.pi * rho_mean_z)
        R = pow(R, 1./3.)
#        print('Scaling R in Mpc: ', R)
#        result = self.cosmo.sigma(R, z)
        # and it seems that R should be R / h
        result = self.cosmo.sigma(R / self.h, z)
#        print('Sigma: ', result)
        return result

    
    def kill_model(self):
        # Clean CLASS
        self.cosmo.struct_cleanup()
        #if you want to completely kill everything then
        self.cosmo.empty()
