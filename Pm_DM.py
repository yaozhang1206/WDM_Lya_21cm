import numpy as np
import matplotlib.pyplot as plt
from classy import Class
from scipy.interpolate import interp1d

"""
This class builds dark matter models, it gives us the P_m needed for the comparison. It also has a useful function to plot the model in comparison to WDM.
Note that one could build CDM by setting m_WDM_keV to go to infinity.
"""


class P_matter(object):
    
    def __init__(self,m_WDM_keV,Omega_CDM,sigma_8):
        # grab the parameters related to matter of WDM candidate, degrees of freedom, and how much WDM
        self.m_WDM_keV = m_WDM_keV
#        self.g_WDM = g_WDM # it doesn't affect the thermal relics
        self.Omega_CDM = Omega_CDM
        self.sigma_8 = sigma_8
        # now time to build class model
        # Define your cosmology (what is not specified will be set to CLASS default parameters, see explanatory.ini for specific information regarding all parameters)
        self.params = {
            'output': 'mPk',
#            'sigma8': 0.8159,
            'sigma8': self.sigma_8,
            'n_s': 0.9667,
            'h': 0.6774,
            'Omega_b': 0.0486,
            'Omega_cdm': self.Omega_CDM,
            'non linear': 'halofit',
            'halofit_min_k_max': 10,
            'z_reio':7.93,
#            'P_k_max_1/Mpc': 10.5,
            'format': 'CLASS',
#            'z_max_pk': 2.25
            'z_max_pk': 4.0
        } # values chosen consistent with planck cosmology, #P_k_max_1/Mpc was 5 before
        # Create an instance of the CLASS wrapper, this class we call for solving Boltzmann equations
        self.cosmo = Class()
        # Set the parameters to the cosmological code, i.e. tells the wrapper to pass this dictionary to the c code
        self.cosmo.set(self.params)
        # Run the whole code, i.e. computes P_m(k,z)
        self.cosmo.compute()
        # Let's make a function to get us our new computed P_m, but first let's get little h
        self.h = self.cosmo.h()
        """ Get rid of this later """
#        k_Mpc_array,P_m_ultra_Mpc_array = np.loadtxt('../def_matterpower_z2.25.dat',unpack=True)
#        self._P_m_ultra = interp1d(k_Mpc_array,P_m_ultra_Mpc_array)
        print('Finished prepping the model for P_m')

    def P_m_CDM_Mpc(self,k_hMpc,z):
        """
        Returns the 3D matter power spectrum obtained from CLASS in units of Mpc^3 (no little h!). Note that this is a function of redshift too.
        
        Inputs: k [h Mpc^-1], z
        
        Outputs: P_m_CDM [Mpc^3]
        """
#        return self.cosmo.pk(k_hMpc * self.h,z) * self.h**3
#        return self.cosmo.pk(k_hMpc,z) * self.h**3
        # so actually pk needs k_Mpc as input and throws P_Mpc, so [Mpc^3] units, no h.
        return self.cosmo.pk_lin(k_hMpc * self.h,z)
#        return self.cosmo.pk(k_hMpc * self.h,z) # makes P1D too high! so use pk_lin instead
    


    def alpha_WDM(self,m_WDM_keV,Omega_CDM):
        """
        Returns the suppression scale due to WDM models free streaming more than CDM, in Mpc.

        Inputs: m_WDM [keV] (mass of the WDM candidate),
                g_WDM (degrees of freedom, i.e. 1.5) but for thermal relics it does nothing, so let's get rid off it
                Omega_DM (dark matter density of the Universe)
                
        Outputs: alpha [Mpc]
        """
        return 0.049 / self.h * pow(self.m_WDM_keV,-1.11) * pow(self.Omega_CDM / 0.25,0.11) * pow(self.h / 0.7,1.22)


    def T_WDM(self,k_Mpc):
        """
        Transfer function of WDM, taken from Viel et al. (~2006), remember that it is dimensionless

        Inputs: k [Mpc^-1], note that no redshift nor little h in the input

        Outputs: T_WDM
        """
        mu = 1.12
        alpha = self.alpha_WDM(self.m_WDM_keV,self.Omega_CDM)
        pa = 1.0 + pow(alpha * k_Mpc, 2 * mu)
        return pow(pa, -5. / mu)

    def P_m_WDM_Mpc(self,k_hMpc,z):
        """
        Returns the 3D matter power spectrum for WDM in units of Mpc^3 (no little h!). It is a function of wavenumber and redshift.
        
        The power spectrum is computed by multiplying the CDM power spectrum by the square of the transfer fuction obtained by Viel et al. (2006?/2005?), i.e. P_WDM = P_CDM * T_WDM^2
        
        Inputs: k [h Mpc^-1], z
        
        Outputs: P_m_WDM [Mpc^3]
        """
        k_Mpc = k_hMpc * self.h
        return self.P_m_CDM_Mpc(k_hMpc,z) * self.T_WDM(k_Mpc) * self.T_WDM(k_Mpc)

    def plot_P_m(self,z):
        """
        Plots the matter power spectrum at redshift zero
        """
        kk = np.logspace(-4,np.log10(5),1000) # in Mpc^{-1}
        Pk_CDM = []
        Pk_WDM = []
        for k in kk:
            Pk_CDM.append(self.P_m_CDM_Mpc(k,z))
            Pk_WDM.append(self.P_m_WDM_Mpc(k,z))
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(kk[0],kk[-1])
        plt.xlabel(r'k  [Mpc$^{-1}$]',fontsize=14)
        plt.title(r'Redshift = '+str(z),fontsize=14)
        plt.ylabel(r'$P(k)$     [Mpc$^3$]',fontsize=14)
        plt.plot(kk,Pk_CDM,'b-',label=r'CDM')
        plt.plot(kk,Pk_WDM,':',color='green',label=r'WDM')
        plt.grid(linestyle='dotted')
        plt.legend(loc='best',fontsize=14)
        plt.show()
        return 'plotted!'
    
    def kill_model(self):
        # Clean CLASS
        self.cosmo.struct_cleanup()
        #if you want to completely kill everything then
        self.cosmo.empty()

#    def P_m_ultra_Mpc(self,k_Mpc):
#        """ for testing Bohua's recent model """
#        return self._P_m_ultra(k_Mpc)
