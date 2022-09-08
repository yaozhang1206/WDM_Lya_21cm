import numpy as np
import Pm_DM as pm # check that the transfer function implementation is the same as Yao's input P
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.integrate import quad
from scipy.special import spherical_jn

"""
    Class in charge of building the theory models for base Lya and memory of reio in Lya with/without WDM.
    
    The class requires a WDM_mass, WDM_21cm and WDM_gadget, i.e. who to call for 21cmFAST results and Gadget transparency table. Given the complexity of the errorbar procedure the nomenclature is the following for a 3 keV WDM mass model calling the gadget realization 1: Gadget_r1_3keV. Note that it would use the average 21cmFAST, so uses wdm_21cm 21cm_ave_3kev. (See errorbar procedure described in Montero-Camacho et al. (2019)
    
    For CDM need to add CDM at the end, so e.g. 21cm_ave_cdm.
    
    In addition, this version accounts for the Pat McDonald modeling of the dependence of b_F and rsd parameter with sigma8, as a result it will need to know which sigma8 value and use the interpolation from 0.7325 to 0.8920.
    
    It is probably a better idea to use the arinyo modeling since we expect ~ 14 % difference or larger if we try to compare different redshifts.
    
    So I need a significant rewrite of everything.

"""

class theory_P_lyas(object):
    # this class basically takes care of my previous code. It computes the theoretical P3D

    def __init__(self,wdm_mass,sigma8,wdm_21cm,wdm_gadget):
        # we will have some hardcoded cosmological parameters here
        """ We need to make sure they, and the ones in Pm_DM, match Yao's input P """
        # hardcoded cosmo
        self.h = 0.6774
        self.omega_m = 0.3088
        self.omega_r = 8.6e-5
        self.omega_l = 1. - self.omega_m # note that I ignore radiation here
        self.omega_b = 0.0486
        self.sigma8 = sigma8
        self.omega_dm = self.omega_m - self.omega_b
        # model details
        self.wdm_mass = wdm_mass # making it a propery of the class in case we want to call it
        self.wdm_21cm = wdm_21cm
        self.wdm_gadget = wdm_gadget
        # construct matter power spectrum, used for computing the conventional lya term
        """ Most likely will have to change this to allow varying dm and sigma8 """
        self.pm = pm.P_matter(self.wdm_mass, self.omega_dm, self.sigma8) # notice I could have called wdm_mass without self here
        # extra details about igm state, we should perhaps invest in not using observational values of these bias
        # file
        self.file_21cm = './data/21cmFAST/cross_'+str(wdm_21cm)+'.txt'
        self.file_gadget = './data/gadget/transp_'+str(wdm_gadget)+'.txt'
        """
        The transparency tables must be in the format of each row correspond to a z_obs from 2.0 to 4.0 and each column is a z_re starting at 6.0 up to 12.0 with an additional final column giving us the bias_G, e.g.
                   # z_re=6    z_re=7                          b_G
        #z_obs=2.    0.06299  0.02590  0.xxx  -0.01460  -0.03112  0.084
        #zobs=2.5    0.07600  0.03307  0.xxx  -0.01865  -0.03723  0.146
            ...
        """
        # defining hyper parameters for the EoR integrals
        self.delta_z = 0.10
        self.N_total = 291
        self.z_obs_array = [2.0, 2.5, 3.0, 3.5, 4.0]
        # setting up some hyper-parameters for correlation
        self.pre_factor_ell0 = 1. / (2. * np.pi**2)
        self.pre_factor_ell2 = -1. / (2. * np.pi**2)
        self.pre_factor_ell4 = self.pre_factor_ell0
        self.k_min = 0.0001
        self.k_max = 10.
        self.n = 5000.
        self.w = (self.k_max - self.k_min) / self.n # width of the intervals
        self.k_trap = np.linspace(self.k_min, self.k_max, int(self.n)+1)
        # and now for the parameters themselves both for nonlinear modeling and biases
        self.z_arinyo = [2.2, 2.4, 2.6, 2.8, 3.0]
        self.sigmas = [0.64, 0.76, 0.8778]
        # betas
        self.beta_rsd_zs = [1.163, 1.284, 1.405, 1.137, 1.257, 1.385, 1.094, 1.219, 1.343, 1.035, 1.161, 1.284, 0.965, 1.086, 1.205]
        self.beta_rsd = interp2d(self.z_arinyo, self.sigmas, self.beta_rsd_zs)
        # bias, this one needs to be converted into a flux bias
        self.b_tau_delta_zs = [0.7490, 0.6401, 0.5536, 0.7478, 0.6428, 0.5574, 0.7429, 0.6409, 0.5588, 0.7366, 0.6373, 0.5577, 0.7287, 0.6319, 0.5546]
        self.b_tau_delta = interp2d(self.z_arinyo, self.sigmas, self.b_tau_delta_zs)
        # non-linear terms
        self.q1_zs = [1.004, 0.804, 0.677, 0.979, 0.783, 0.666, 0.958, 0.767, 0.652, 0.949, 0.758, 0.644, 0.958, 0.762, 0.648]
        self.q1 = interp2d(self.z_arinyo, self.sigmas, self.q1_zs)
        # next
        self.kp_hMpc_zs = [15.3, 14.5, 13.3, 15.1, 14.6, 13.5, 14.7, 14.4, 13.6, 14.2, 14.1, 13.4, 13.6, 13.7, 13.1]
        self.kp_hMpc = interp2d(self.z_arinyo, self.sigmas, self.kp_hMpc_zs)
        # kv
        self.kv_hMpc_to_av_zs = [1.677, 1.211, 0.961, 1.772, 1.242, 0.963, 1.879, 1.270, 0.970, 2.015, 1.314, 0.979, 2.191, 1.383, 1.007]
        self.kv_hMpc_to_av = interp2d(self.z_arinyo, self.sigmas, self.kv_hMpc_to_av_zs)
        # av
        self.av_zs = [0.818, 0.659, 0.533, 0.868, 0.700, 0.561, 0.915, 0.734, 0.590, 0.958, 0.763, 0.610, 0.994, 0.788, 0.627]
        self.av = interp2d(self.z_arinyo, self.sigmas, self.av_zs)
        # bv
        self.bv_zs = [1.53, 1.55, 1.54, 1.56, 1.58, 1.58, 1.59, 1.61, 1.61, 1.61, 1.64, 1.64, 1.63, 1.66, 1.66]
        self.bv = interp2d(self.z_arinyo, self.sigmas, self.bv_zs)
        # later we are going to need a reference redshift
        self.z_ref = 3.0
        # call constructor
        self._setup_theory()
        
    def log_observed_Flux(self, z):
        """ Mean observed flux, useful to get flux bias """
        return -0.0023 * pow((1. + z), 3.65)
        
    def flux_bias(self, z, sigma8):
        """ Obtains the flux bias from arinyo tables """
        if z<= 3.0:
            return self.b_tau_delta(z, sigma8) * self.log_observed_Flux(z)
        else:
            return self.b_tau_delta(3.0, sigma8) * self.log_observed_Flux(3.0)
            
    def deldel_lin(self, z, k_Mpc):
        """ Dimensionless linear power density, for the non-linear correction """
        k_hMpc = k_Mpc / self.h # careful here, I did this wrong initially.
        if z<= 3.0:
            return pow(k_Mpc, 3.0) * self.pm.P_m_WDM_Mpc(k_hMpc, z) / (2. * np.pi**2)
        else:
            return pow(k_Mpc, 3.0) * self.pm.P_m_WDM_Mpc(k_hMpc, 3.0) / (2. * np.pi**2)
            
    def D_Mpc_Arinyo2015(self,z,k_Mpc,mu,sigma8):
        """ Analytic formula for the nonlinear correction to the 3D lya power spectrum from Arinyo 2015 """
        if z <=3.0:
            k_p = self.kp_hMpc(z,sigma8) * self.h
            k_v_to_av = self.kv_hMpc_to_av(z,sigma8) # the hMpc is not true here actually !
            a_v = self.av(z,sigma8)
            b_v = self.bv(z,sigma8)
            q_1 = self.q1(z,sigma8)
        else:
            k_p = self.kp_hMpc(3.0,sigma8) * self.h
            k_v_to_av = self.kv_hMpc_to_av(3.0,sigma8)  # also not true! we handle that in m_factor
            a_v = self.av(3.0,sigma8)
            b_v = self.bv(3.0,sigma8)
            q_1 = self.q1(3.0,sigma8)
        p_factor = pow(k_Mpc / k_p,2.0)
        # in what follows we handle the problem with the little h fit included in the k_v
        m_factor = 1. - pow(k_Mpc / self.h,a_v) / k_v_to_av * pow(mu,b_v)
        temp = self.deldel_lin(z,k_Mpc)
        nl_factor_WDM = q_1 * temp
        c_unt_WDM = nl_factor_WDM * m_factor - p_factor
        return np.exp(c_unt_WDM)
        
    def FluxP3D_lya_Mpc(self, z, k, mu):
        """ 3D Base Lya flux power spectrum with redshift evolution (after z=3) with a trick, k must be in inverse Mpc. """
        bias = self.flux_bias(z, self.sigma8)
        k_hMpc = k / self.h
        if z <= 3.0:
            kaiser = pow(bias * (1.0 + self.beta_rsd(z, self.sigma8) * pow(mu, 2)), 2)
            P3D = self.pm.P_m_WDM_Mpc(k_hMpc, z) * kaiser
        else:
            kaiser = pow(bias * (1.0 + self.beta_rsd(3.0, self.sigma8) * pow(mu, 2)), 2)
            z_evol = pow((1.0 + z) / (1.0 + self.z_ref), 3.55) # got this from https://www.aanda.org/articles/aa/pdf/2013/11/aa22130-13.pdf
            P3D = self.pm.P_m_WDM_Mpc(k_hMpc, 3.0) * kaiser * z_evol
        return P3D

    def nonlinear_FluxP3D_lya_Mpc(self, z, k, mu, linear=False):
        """ 3D Base Lya flux power spectrum with redshift evolution (after z=3) with a trick, k must be in inverse Mpc. """
        bias = self.flux_bias(z, self.sigma8)
        k_hMpc = k / self.h
        if z <= 3.0:
            kaiser = pow(bias * (1.0 + self.beta_rsd(z, self.sigma8) * pow(mu, 2)), 2)
            P3D = self.pm.P_m_WDM_Mpc(k_hMpc, z) * kaiser
        else:
            kaiser = pow(bias * (1.0 + self.beta_rsd(3.0, self.sigma8) * pow(mu, 2)), 2)
            z_evol = pow((1.0 + z) / (1.0 + self.z_ref), 3.55) # got this from https://www.aanda.org/articles/aa/pdf/2013/11/aa22130-13.pdf
            P3D = self.pm.P_m_WDM_Mpc(k_hMpc, z) * kaiser * z_evol
        if linear:
            return P3D
        else:
            # nonlinear correction
            cor = self.D_Mpc_Arinyo2015(z, k, mu, self.sigma8)
            return P3D * cor
        
    # defining some useful functions
    def hubble(self,z):
        # returns the hubble parameter as a function of z for a given cosmology for the model
        return (self.h*100)*np.sqrt(self.omega_m*(1 + z)**3 + self.omega_r*(1 + z)**4 + (1.0 - self.omega_m))

    def integrand_D(self,x):
        return 1.0 * (1.0 + x) / (1.0 * self.hubble(x))**3
            
    def Dratio(self,z_obs, z):
        D_obs = quad(self.integrand_D, z_obs, np.inf)
        D = quad(self.integrand_D, z, np.inf)
        return (1.0 * D_obs[0] * self.hubble(z_obs)) / (1.0 * D[0] * self.hubble(z))
        
    # besides I need to do the integration over the EoR, doing a simple riemann sum then
    def crosspower_z1(self, z, k):
        result = 0. # reset
        j = 0
        for j in range(0,self.N_total):
            result = result + interpolate.splev(z, self.tck_z1, der=1) * self.cross(z, k) * (2.0 * np.pi**2 / (1.0 * k**3)) * self.Dratio(self.z_obs_array[0], z) * self.delta_z
            # note that this is a P and not a density!
            z = z + self.delta_z
        return result * (-1.)
    
    def crosspower_z2(self, z, k):
        result = 0. # reset
        j = 0
        for j in range(0,self.N_total):
            result = result + interpolate.splev(z, self.tck_z2, der=1) * self.cross(z, k) * (2.0 * np.pi**2 / (1.0 * k**3)) * self.Dratio(self.z_obs_array[1], z) * self.delta_z
            # note that this is a P and not a density!
            z = z + self.delta_z
        return result * (-1.)
        
    def crosspower_z3(self, z, k):
        result = 0. # reset
        j = 0
        for j in range(0,self.N_total):
            result = result + interpolate.splev(z, self.tck_z3, der=1) * self.cross(z, k) * (2.0 * np.pi**2 / (1.0 * k**3)) * self.Dratio(self.z_obs_array[2], z) * self.delta_z
            # note that this is a P and not a density!
            z = z + self.delta_z
        return result * (-1.)
        
    def crosspower_z4(self, z, k):
        result = 0. # reset
        j = 0
        for j in range(0,self.N_total):
            result = result + interpolate.splev(z, self.tck_z4, der=1) * self.cross(z, k) * (2.0 * np.pi**2 / (1.0 * k**3)) * self.Dratio(self.z_obs_array[3], z) * self.delta_z
            # note that this is a P and not a density!
            z = z + self.delta_z
        return result * (-1.)
        
    def crosspower_z5(self, z, k):
        result = 0. # reset
        j = 0
        for j in range(0,self.N_total):
            result = result + interpolate.splev(z, self.tck_z5, der=1) * self.cross(z, k) * (2.0 * np.pi**2 / (1.0 * k**3)) * self.Dratio(self.z_obs_array[4], z) * self.delta_z
            # note that this is a P and not a density!
            z = z + self.delta_z
        return result * (-1.)
        
    def _setup_theory(self):
        # constructor in charge of making the computations
        # grab large-scale info
        self.z_21cm, self.k_21cm, self.P_mxHI = np.loadtxt(self.file_21cm, unpack=True)
        self.z_21cm = np.unique(self.z_21cm)
        self.k_21cm = np.unique(self.k_21cm)
        # interpolate cross-power
        self.cross = interp2d(self.z_21cm, self.k_21cm, self.P_mxHI)
        # now do the small-scale info
        igm_table = np.loadtxt(self.file_gadget)
        # separate for simplicity
        y_z1 = igm_table[0,:-1]
        y_z2 = igm_table[1,:-1]
        y_z3 = igm_table[2,:-1]
        y_z4 = igm_table[3,:-1]
        y_z5 = igm_table[4,:-1]
        # also grabing the bias, let's rename for clarity
        bias_G_array = igm_table[:,-1]
        # interpolate radiation bias
        self.G_bias = interp1d(self.z_obs_array, bias_G_array, fill_value="extrapolate")
        # and the redshifts
        x = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=float)
        self.tck_z1 = interpolate.splrep(x, y_z1, k=1, s=0)
        self.tck_z2 = interpolate.splrep(x, y_z2, k=1, s=0)
        self.tck_z3 = interpolate.splrep(x, y_z3, k=1, s=0)
        self.tck_z4 = interpolate.splrep(x, y_z4, k=1, s=0)
        self.tck_z5 = interpolate.splrep(x, y_z5, k=1, s=0)
        # build the transparency power spectrum
        crosspower_psi_array = np.zeros(len(self.k_21cm) * len(self.z_obs_array))
        a = 0
        b = 0
        for a in range(0,5):
            b = 0 # reset
            for b in range(0,len(self.k_21cm)):
                if a==0:
                    crosspower_psi_array[b] = self.crosspower_z1(5.90, self.k_21cm[b])
                elif a==1:
                    crosspower_psi_array[b+len(self.k_21cm)] = self.crosspower_z2(5.90,self.k_21cm[b])
                elif a==2:
                    crosspower_psi_array[b+len(self.k_21cm)*2] = self.crosspower_z3(5.90,self.k_21cm[b])
                elif a==3:
                    crosspower_psi_array[b+len(self.k_21cm)*3] = self.crosspower_z4(5.90,self.k_21cm[b])
                elif a==4:
                    crosspower_psi_array[b+len(self.k_21cm)*4] = self.crosspower_z5(5.90,self.k_21cm[b])
        # time to interpolate
        self._crosspower_psi = interp2d(self.z_obs_array, self.k_21cm, crosspower_psi_array, kind='cubic')
        # for the bias for large scales
        k_lows = [5.263151e-02, 7.314235e-02, 1.025928e-01]
        bias_re_array_z1 = np.zeros(len(k_lows))
        bias_re_array_z2 = np.zeros(len(k_lows))
        bias_re_array_z3 = np.zeros(len(k_lows))
        bias_re_array_z4 = np.zeros(len(k_lows))
        bias_re_array_z5 = np.zeros(len(k_lows))
        for i in range(0, len(k_lows)):
            bias_re_array_z1[i] = self._crosspower_psi(2.0, k_lows[i]) / self.pm.P_m_WDM_Mpc(k_lows[i] / self.h, 2.0)
            bias_re_array_z2[i] = self._crosspower_psi(2.5, k_lows[i]) / self.pm.P_m_WDM_Mpc(k_lows[i] / self.h, 2.5)
            bias_re_array_z3[i] = self._crosspower_psi(3.0, k_lows[i]) / self.pm.P_m_WDM_Mpc(k_lows[i] / self.h, 3.0)
            bias_re_array_z4[i] = self._crosspower_psi(3.5, k_lows[i]) / self.pm.P_m_WDM_Mpc(k_lows[i] / self.h, 3.5)
            bias_re_array_z5[i] = self._crosspower_psi(4.0, k_lows[i]) / self.pm.P_m_WDM_Mpc(k_lows[i] / self.h, 4.0)
        bias_re_append = np.append(bias_re_array_z1, bias_re_array_z2)
        bias_re_append = np.append(bias_re_append, bias_re_array_z3)
        bias_re_append = np.append(bias_re_append, bias_re_array_z4)
        bias_re_append = np.append(bias_re_append, bias_re_array_z5)
        # interpolate
        self._bias_re = interp2d(self.z_obs_array, k_lows, bias_re_append)

    def psi_power(self, z, k_Mpc):
        """ using this to include the biasing procedure """
        k_min = 7.314235e-02
        k_max = 10.
        k_cut = 5.263151e-02
        if k_Mpc <= k_max and k_Mpc >= k_min:
            return self._crosspower_psi(z, k_Mpc)
        elif k_Mpc > k_max:
            return 0. # force to disapear at small scales
        elif k_Mpc >= k_cut and k_Mpc < k_min:
            bias_scaling = self._bias_re(z, k_Mpc)
            return bias_scaling * self.pm.P_m_WDM_Mpc(k_Mpc / self.h, z)
        elif k_Mpc < k_cut:
            bias_scaling = self._bias_re(z, k_cut)
            return bias_scaling * self.pm.P_m_WDM_Mpc(k_Mpc / self.h, z)

    def bias_reio(self, z, k_Mpc, mu):
        """ need to include annyoing redshift evolution of the biases in reionization term """
        bias = self.flux_bias(z, self.sigma8)
        if z <= 3.0:
            return bias * self.G_bias(z) * (1. + self.beta_rsd(z, self.sigma8) * mu**2)
        else:
            temp = bias * self.G_bias(z) * (1. + self.beta_rsd(3.0, self.sigma8) * mu**2)
            z_evol = pow((1.0 + z) / (1.0 + self.z_ref), 3.55)
            return temp * z_evol * np.sqrt(self.pm.P_m_WDM_Mpc(k_Mpc / self.h, 3.0) / (1. * self.pm.P_m_WDM_Mpc(k_Mpc / self.h, z)))

    def FluxP3D_reio_Mpc(self, z, k, mu):
        """ Returns the 3D memory of reionization in the lya forest in Mpc^3, unit for k is Mpc^-1. It has a cutoff just in case """
        bias = self.bias_reio(z, k, mu)
        return 2. * bias * self.psi_power(z, k)

    # making things easier to plot IGM plot
    def transparency(self, z_re,z_obs):
        """ returns a given z_obs psi(z_re) """
        if z_obs == 2.0:
            return interpolate.splev(z_re, self.tck_z1, der=0)
        elif z_obs == 3.0:
            return interpolate.splev(z_re, self.tck_z3, der=0)
        elif z_obs == 4.0:
            return interpolate.splev(z_re, self.tck_z5, der=0)
        else:
            print('Come here and code it yourself, since I only did two redshifts :)')


    """ Need to deal with the correlation function multipoles, solution is to deal with redshift <= 3 """

    def P_mu0_Mpc(self, z, k_Mpc, reio):
        """ Analytic formula for the mu^0 term in Mpc^3 """
        P = self.flux_bias(z, self.sigma8)**2 * self.pm.P_m_WDM_Mpc(k_Mpc / self.h, z)
        if reio == False:
            return P
        else:
            P_reio = 2. * self.flux_bias(z, self.sigma8) * self.G_bias(z) * self.psi_power(z, k_Mpc)
            return P + P_reio
    
    def P_mu2_Mpc(self, z, k_Mpc, reio):
        """ Analytic formula for the quadrupole term in Mpc^3 """
        P = self.flux_bias(z, self.sigma8)**2 * self.beta_rsd(z, self.sigma8) * self.pm.P_m_WDM_Mpc(k_Mpc / self.h, z)
        if reio == False:
            return P
        else:
            P_reio = self.flux_bias(z, self.sigma8) * self.G_bias(z) * self.beta_rsd(z, self.sigma8) * self.psi_power(z, k_Mpc)
            return P + P_reio
        
    def P_mu4_Mpc(self, z, k_Mpc):
        """ Analytic formula for the hexadecapole in Mpc^3 """
        P = self.flux_bias(z, self.sigma8)**2 * self.beta_rsd(z, self.sigma8)**2 * self.pm.P_m_WDM_Mpc(k_Mpc / self.h, z)
        return P
    
    # and now for the correlation function
    def integ_ell0(self, k_Mpc, r_Mpc, z):
        # no-reio monopole
        P_ell0 = self.P_mu0_Mpc(z, k_Mpc, False) + 2. * self.P_mu2_Mpc(z, k_Mpc, False) / 3. + self.P_mu4_Mpc(z, k_Mpc) / 5.
        return pow(k_Mpc, 2) * spherical_jn(0, k_Mpc * r_Mpc, False) * P_ell0
    
    def reio_integ_ell0(self, k_Mpc, r_Mpc, z):
        # include reio monopole
        reio_P_ell0 = self.P_mu0_Mpc(z, k_Mpc, True) + 2. * self.P_mu2_Mpc(z, k_Mpc, True) / 3. + self.P_mu4_Mpc(z, k_Mpc) / 5.
        return pow(k_Mpc, 2) * spherical_jn(0, k_Mpc * r_Mpc, False) * reio_P_ell0
    
    def integ_ell2(self, k_Mpc, r_Mpc, z):
        # no-reio quadrupole
        P_ell2 = 4. * self.P_mu2_Mpc(z, k_Mpc, False) / 3. + 4. * self.P_mu4_Mpc(z, k_Mpc) / 7.
        return pow(k_Mpc, 2) * spherical_jn(2, k_Mpc * r_Mpc, False) * P_ell2
        
    def reio_integ_ell2(self, k_Mpc, r_Mpc, z):
        # include reio quadrupole
        reio_P_ell2 = 4. * self.P_mu2_Mpc(z, k_Mpc, True) / 3. + 4. * self.P_mu4_Mpc(z, k_Mpc) / 7.
        return pow(k_Mpc, 2) * spherical_jn(2, k_Mpc * r_Mpc, False) * reio_P_ell2
    
    def integ_ell4(self, k_Mpc, r_Mpc, z):
        # only component
        P_ell4 = 8. * self.P_mu4_Mpc(z, k_Mpc) / 35.
        return pow(k_Mpc, 2) * spherical_jn(4, k_Mpc * r_Mpc, False) * P_ell4
        
    # setting up trapezoidal, it is kind of annoying so combine into a single function
    def correlation_components_Mpc(self, r_Mpc, z):
        # integrate and it throws them in order: ell0, ell2, ell4, reio_ell0, reio_ell2
        ell0 = np.zeros(len(self.k_trap))
        ell2 = np.zeros(len(self.k_trap))
        ell4 = np.zeros(len(self.k_trap))
        reio_ell0 = np.zeros(len(self.k_trap))
        reio_ell2 = np.zeros(len(self.k_trap))
        for i in range(0, len(self.k_trap)):
            ell0[i] = self.integ_ell0(self.k_trap[i], r_Mpc=r_Mpc, z=z)
            ell2[i] = self.integ_ell2(self.k_trap[i], r_Mpc=r_Mpc, z=z)
            ell4[i] = self.integ_ell4(self.k_trap[i], r_Mpc=r_Mpc, z=z)
            reio_ell0[i] = self.reio_integ_ell0(self.k_trap[i], r_Mpc=r_Mpc, z=z)
            reio_ell2[i] = self.reio_integ_ell2(self.k_trap[i], r_Mpc=r_Mpc, z=z)
        result_ell0 = self.w * (ell0.sum() - (ell0[0] + ell0[-1]) / 2.)
        result_ell2 = self.w * (ell2.sum() - (ell2[0] + ell2[-1]) / 2.)
        result_ell4 = self.w * (ell4.sum() - (ell4[0] + ell4[-1]) / 2.)
        result_reio_ell0 = self.w * (reio_ell0.sum() - (reio_ell0[0] + reio_ell0[-1]) / 2.)
        result_reio_ell2 = self.w * (reio_ell2.sum() - (reio_ell2[0] + reio_ell2[-1]) / 2.)
        result_ell0 = result_ell0 * self.pre_factor_ell0
        result_ell2 = result_ell2 * self.pre_factor_ell2
        result_ell4 = result_ell4 * self.pre_factor_ell4
        result_reio_ell0 = result_reio_ell0 * self.pre_factor_ell0
        result_reio_ell2 = result_reio_ell2 * self.pre_factor_ell2
        return result_ell0, result_ell2, result_ell4, result_reio_ell0, result_reio_ell2

    # the aliasing term will require P1D, we will use the empirical formula since here we only care about what the memory
    # can do by itself
    def P1D_z_kms_PD2013(self,z,k_kms):
        """ Fitting formula from Palanque-Delabrouille et al. (2013). Note the funny units for both k and P. This can be used for comparison too """
        # this function is mainly for comparison purposes since it does not play well with a fisher forecast
        A_F = 0.064
        n_F = -2.55
        alpha_F = -0.1
        B_F = 3.55
        beta_F = -0.28
        k0 = 0.009
        z0 = 3.0
        n_F_z = n_F + beta_F * np.log((1.0 + z) / (1.0 + z0))
        # correction to cut low k behaviour
        k_min = k0*np.exp((-0.5 * n_F_z - 1.0) / alpha_F)
        k_kms = np.fmax(k_kms,k_min)
        exp1 = 3.0 + n_F_z + alpha_F * np.log(k_kms / k0)
        toret = np.pi * A_F / k0 * pow(k_kms / k0, exp1 - 1.0) * pow((1.0 + z) / (1.0 + z0), B_F)
        return toret
    
    
