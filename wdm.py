import numpy as np
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg
import sys
# import puma
# import skalow
from math import floor
import pickle

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
    
    def __init__(self, k, mu, z, params, param_shift, t_int, bandwidth, exp_name):
    
        self.k = k
        self.mu = mu
        self.z = z
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
        self.bHI = params['bHI']
        self.OHI = params['O_HI'] / 1.e3
        # self.OHI = self.OHI(self.z)
        self.param_shift = param_shift
        self.t_int = t_int # 21 cm IM observation time in units of hr
        self.bandwidth = bandwidth # 21 cm experiment bandwidth, in units of MHz. 300 for SKA, 200 for PUMA
        self.exp_name = exp_name

        self.OMh2 = self.Obh2 + self.Och2
        self.pk()
        
    # def OHI(self,z):
    #     if 3.49 < z < 4.45:
    #         return 1.18e-3
    #     elif 4.44 < z < 5.51:
    #         return 0.98e-3

    def Tb_mean(self):
        '''
        output in units of mK
        '''
        
        return 27*np.sqrt((1+self.z)/10*0.15/self.OMh2)*(self.OHI*self.h**2/0.023)


    def pk(self):
        '''
        output is a function P(k), k in units of Mpc^{-1}, P(k) is in units of Mpc^3
        need to determine file names
        '''

        pk = []
        zs = np.arange(3.5,6.0,0.5)
        for z in range(5):
            # z from 5.5 to 3.5, files numbered 1-5
            data=np.loadtxt('./patchy_reion_ns/%s/z%d_pk.dat' % (self.param_shift,-z+5))
            pk.append(data[:,1])
        k = data[:,0]
        pk = np.array(pk) / self.h**3
#         pk *= np.array(k)**3/2/np.pi**2 
        coords = []
        for i in zs:
            for j in k:
                coords.append((i,j * self.h))
            
        self.pk_fun = interpolate.LinearNDInterpolator(coords, pk.flatten())

    def f(self):
    # Growth rate = OM(z)^0.545 = (OM*(1+z)^3/(OM*(1+z)^3+(1-OM)))^0.545 ref: 1709.07893
        OM = (self.Obh2+self.Och2)/self.h**2
        return (OM*(1+self.z)**3/(OM*(1+self.z)**3+1-OM))**0.545


    def P_fid(self):
        '''
        Output units mK^2
        '''
        return self.Tb_mean()**2 * (self.bHI + self.mu**2*self.f())**2 * self.pk_fun(self.z,self.k)
    
    def PN(self,t_int,b):
        '''
        Input: t_int is 21cm IM mission time in units of hour, b is the bandwidth in MHz
        Output units mK^2 Mpc^3
        '''
#         t_int = 24*365*5
#         b = 8

        
        Obh2 = 0.02230
        Och2 = 0.1188
        h = 0.6774

        OM = (Obh2+Och2)/h**2
        Or = 0
        if self.exp_name == 'SKA':
            noise = skalow.skalow(t_int, b, OM, h,Or)
        elif self.exp_name == 'PUMA':
            noise = puma.puma(t_int, b, OM, h,Or)
        PN = noise.noise_power_Mpc(self.z,self.k,self.mu)
        print('At k = %f, mu = %f, z=%f, param_scenario = %s, P_N = %f '% (self.k, self.mu, self.z, self.param_shift, PN))

        
        return PN
    
    def P_21obs(self):
        # if self.experiment == 'SKA':
        #     t_int = 100
        #     b = 32
        # elif self.experiment == 'PUMA':
        #     t_int = 24*365*5
        #     b = 200
        P_fid = self.P_fid()+self.PN(self.t_int, self.bandwidth)
        print('At k = %f, mu = %f, z=%f, param_scenario = %s, P_fid = %f '% (self.k, self.mu, self.z, self.param_shift, P_fid))

        return P_fid
    

class P_patchy_reion:
    
    def __init__(self, k, mu, z, params, reion_his_file, cross_ps_file, verbose = True):
    
        self.k = k
        self.mu = mu
        self.z = z
        self.h = params['h']
        # self.h = 0.6774
        self.Obh2 = params['Obh2']
        self.Och2 = params['Och2']
        self.mnu2 = params['mnu']
        self.As = params['As'] / 1.e9
        self.ns = params['ns']
        self.alpha_s = params['alphas']
        self.tau_re = params['taure']
        self.bHI = params['bHI']
        self.OHI = params['O_HI'] / 1.e3
        # self.OHI = self.OHI(self.z)
        self.reion_his_file = reion_his_file
        self.cross_ps_file = cross_ps_file
        self.verbose = verbose

        self.OMh2 = self.Obh2+self.Och2
        

        # self.param_shift = param_shift

        self.delta_crit=1.686 # critcal overdensity for collapse
        self.Mhalos=np.logspace(7,16,10000)
        self.ks=np.logspace(-4,2,1000)
        
        G=4.30091e-9
        rho_crit=3*(self.h*100)**2/8/np.pi/G*self.h # M_solar/(Mpc)^3/h
        self.rhom=rho_crit*self.OMh2/self.h**2 #*(1+z)**3
        
#         print(self.rhom)

        
    # def OHI(self,z):
    #     if 3.49 < z < 4.45:
    #         return 1.18e-3
    #     elif 4.44 < z < 5.51:
    #         return 0.98e-3
        
    def Tb_mean(self):
        
        return 27*np.sqrt((1+self.z)/10*0.15/self.OMh2)*(self.OHI*self.h**2/0.023)

    def reion_his(self,filename):
        # data = np.loadtxt(filename)
        data = np.loadtxt(filename, delimiter=',')
        z_rh =data.T[0]
        xH = data.T[1]
        return interpolate.interp1d(z_rh,xH)
        

    def Pk(self):
        '''
        output is a function P(k), k in units of Mpc^{-1}, P(k) is in units of Mpc^3
        need to determine file names
        '''

        pk = []
        zs = np.arange(3.5,6.0,0.5)
        for z in range(5):
            # z from 5.5 to 3.5, files numbered 1-5
            data=np.loadtxt('/Users/heyang/Dropbox/Research/reion_ns/codes/Analysis/patchy_reion_ns/fid/z%d_pk.dat' % (-z+5))
            pk.append(data[:,1])
        k = data[:,0]
        pk = np.array(pk) / self.h**3
#         pk *= np.array(k)**3/2/np.pi**2 
        coords = []
        for i in zs:
            for j in k:
                coords.append((i,j* self.h))
            
        self.pk_fun = interpolate.LinearNDInterpolator(coords, pk.flatten())
        return self.pk_fun


    def halo_func(self):
        pk_fun=self.Pk()

        A=0.186
        a=1.47
        b=2.57
        c=1.19
        
        sigmas=[]
#         print(pk_fun(self.z,self.ks))
        
        for m in self.Mhalos:
            R=(3.*m/4./np.pi/self.rhom)**(1./3)
            w_kR=3./(self.ks*R)**3*(np.sin(self.ks*R)-self.ks*R*np.cos(self.ks*R))
            d_sigma2=pk_fun(self.z,self.ks)*w_kR**2*self.ks**2/(2*np.pi**2)
#             print(d_sigma2)
            sigma_2=integrate.simps(d_sigma2,self.ks)
#             print(sigma_2)
            sigma=np.sqrt(sigma_2)  
            sigmas.append(sigma)

        sigmas=np.array(sigmas)
        self.sigmas=sigmas
        f_sigma=A*((sigmas/b)**(-a)+1)*np.exp(-c/sigmas**2)

        self.dndM=f_sigma*self.rhom/self.Mhalos*np.gradient(np.log(sigmas**(-1)),self.Mhalos)

    def f(self):
    # Growth rate = OM(z)^0.545 = (OM*(1+z)^3/(OM*(1+z)^3+(1-OM)))^0.545 ref: 1709.07893
        OM = self.OMh2/self.h**2

        return (OM*(1+self.z)**3/(OM*(1+self.z)**3+1-OM))**0.545
        
    def M_b(self, Mhalo, z_re, his_file, fmass_file, z_obs):
        f_b = 0.1573
        xh_arr = self.reion_his(his_file)
        MF=pd.read_csv(fmass_file,index_col='z_obs')
        z_res=[6.0,7.0,8.0,9.0,10.0,11.0,12.0]

        if z_re==6.0: 
            Mb = f_b*Mhalo*(1+(2**(1./3)-1)*MF.loc[z_obs][0]/Mhalo)**(-3.)
        else: 
            z_range=np.arange(6.0,z_re+0.01,0.01)
            xh_dot=np.gradient(xh_arr(z_range),z_range)
            mf = interpolate.interp1d(np.array(z_res),MF.loc[z_obs].to_numpy(),fill_value="extrapolate")
            Mb = integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf(z_range)/Mhalo)**(-3.)*xh_dot,z_range)

        return Mb

    def calc_rho_HI(self, fmass_file, history_file):
        self.halo_func()
        xHI_arr = self.reion_his(history_file)

        f_b = 0.1573
        MF=pd.read_csv(fmass_file,index_col='z_obs')
        mfs = MF.to_numpy().T
        M_HIs = f_b*Mhalo*(1+(2**(1./3)-1)*mfs/Mhalo)**(-3.)

        rho_HI=[]  
        # z_obs = np.arange(3.5,5.6,0.1)
        z_obs = MF.index
        coords = []
        
        z_range=np.arange(6.0,12.01,0.01)
        xHI = xi_arr(z_range)
        xHI[0] = 0
        xHI[-1] = 1
        xHI_dot=np.gradient(xHI,z_range)


        for zobs in z_obs:
            mf = interpolate.interp1d(np.array(z_res),MF.loc[zobs].to_numpy(),fill_value="extrapolate")
            M_baryon = np.array([integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf(z_range)/Mhalo)**(-3.)*xHI_dot,z_range) for Mhalo in self.Mhalos])

        for z_re in z_res:
            for zobs in z_obs:
                print('z_re = %.1f, z_obs = %.1f' % (z_re, zobs))
                # M_baryon = np.array([self.M_b(mh, z_re, history_file, fmass_file, zobs)[0] for mh in self.Mhalos])
                # mf = MF.loc[round(zobs,1)][str(z_re)]
                # rho_HI.append(integrate.simps(self.dndM*M_baryon, self.Mhalos))
                coords.append((z_re,zobs))
                if z_re<=6.0: 
                    M_baryon = np.array([f_b*Mhalo*(1+(2**(1./3)-1)*MF.loc[zobs][0]/Mhalo)**(-3.) for Mhalo in self.Mhalos])
                else: 
                    z_range=np.arange(6.0,z_re+0.01,0.01)
                    xi_dot=np.gradient(xi_arr(z_range),z_range)
                    # print(integrate.simps(xi_dot,z_range))
                    # xi_dot_nv=np.gradient(xi_arr_nov(z_range),z_range)
                    mf = interpolate.interp1d(np.array(z_res),MF.loc[zobs].to_numpy(),fill_value="extrapolate")
                    M_baryon = np.array([integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf(z_range)/Mhalo)**(-3.)*xi_dot,z_range) for Mhalo in self.Mhalos])
                    # M_baryon = np.array([self.M_b(mh, z_re, history_file, fmass_file, zobs) for mh in self.Mhalos])
                rho_HI.append(integrate.simps(self.dndM*M_baryon, self.Mhalos))
                Mb.append(M_baryon)

        rho_HI = interpolate.LinearNDInterpolator(coords, np.array(rho_HI).flatten())
        return rho_HI, np.array(Mb).reshape(len(z_res),-1)
    
    def rho_HI(self):
        # z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]
        # z_obs = np.arange(3.5,5.6,0.1)

        # coords = []
        
        # for z_re in z_res:
        #     for zobs in z_obs:               
        #         xi = bias_v3.b_sink(z_re, zobs, 'fid')
        #         xi_sv, xi_nv = xi.xi_func()
        #         rho = bias_v3.bias_v(z_re, zobs, 'fid')
        #         M_baryon = np.array([rho.M_b(mh,z_re,'fmass_mean.csv',xi_sv)[0] for mh in self.Mhalos])
        #         rho_HI.append(integrate.simps(self.dndM*M_baryon,self.Mhalos))
        #         coords.append((z_re,zobs))
        # rho_HI = interpolate.LinearNDInterpolator(coords, np.array(rho_HI).flatten())


        with open('./rho_HI_func.pkl','rb') as f:
        # with open(self.rho_HI_func_file,'rb') as f:
           rho_HI = pickle.load(f)
           
        return rho_HI

    def rho_HI_ext(self):
        z_res=[5.75, 6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]
        z_obs_arr = np.arange(3.5,5.6,0.1)


        rho_HI_func = self.rho_HI()
        rho_HI = []
        coords = []

        z_re_1 = 6.
        z_re_2 = 7
        for z_re in z_res:
            for z_obs in z_obs_arr:
                if z_re < 6:
                    rho_HI.append(rho_HI_func(z_re_1,z_obs)+(z_re-z_re_1)/(z_re_2-z_re_1)*(rho_HI_func(z_re_2, z_obs)-rho_HI_func(z_re_1, z_obs)))
                else:
                    rho_HI.append(rho_HI_func(z_re, z_obs))

                coords.append((z_re,z_obs))

        rho_HI = interpolate.LinearNDInterpolator(coords, np.array(rho_HI).flatten())
        return rho_HI


    def reion_mid(self,file):
        xi_arr = self.reion_his(file)
        for z in np.arange(9.0,6.0,-0.01):
            if (xi_arr(z)<0.5 and xi_arr(z+0.01)>0.5): break

        z = z + ((0.5-(xi_arr(z)))/(xi_arr(z+0.01)-xi_arr(z)))*0.01

        return z

    def psi(self,z_re,z_obs):
        z_re_mean = self.reion_mid(self.reion_his_file)
        rho_HI_func = self.rho_HI_ext()
        # if z_re >= 6.0:
        #     rho_HI = rho_HI_func(z_re, z_obs)
        # else:
        #     rho_HI = self.rho_HI_ext(z_re, z_obs)
        
        psi = np.log(rho_HI_func(z_re, z_obs)/rho_HI_func(z_re_mean, z_obs))


    def dpsi_dz(self):
        zmin = 5.5
        zmax = 12.0
        z_res = np.arange(zmin,zmax+0.1,0.01)
        z_re_mean = self.reion_mid(self.reion_his_file)
#         print(z_re_mean)
        
        
        rho_HI = self.rho_HI_ext()
#         print(rho_HI(z_res,self.z),rho_HI(z_re_mean,self.z))
    
        dpsi_dz = np.gradient(np.log(rho_HI(z_res,self.z)/rho_HI(z_re_mean,self.z)),z_res)
        dpsi_dz[np.isnan(dpsi_dz)]=0

        return dpsi_dz

    def PmxH(self,cross_ps_file):
        zmin = 5.5
        zmax = 12.0
        z_res = np.arange(zmin,zmax+0.1,0.01)
        Pc = np.loadtxt(cross_ps_file)

        # original Pk in dimensionless, transform to Mpc^3
        Pk = Pc.T[2] * 2 * np.pi**2 / Pc.T[1]**3
        # Pk = Pc.T[2]

        # interpolate Pk on (z,k) plane
        P_m_xH_func = interpolate.LinearNDInterpolator(Pc[:,0:2], np.array(Pk).flatten())
        
        PmxH = P_m_xH_func(z_res,self.k)

        return PmxH

    def P_m_psi(self):
        zmin = 5.5
        zmax = 12.0
        z_res = np.arange(zmin,zmax+0.1,0.01)
 
        dpsi_dz = self.dpsi_dz()

        PmxH = self.PmxH(self.cross_ps_file)
        
        P_m_psi = -integrate.simps(dpsi_dz*PmxH*(np.ones(z_res.shape)+z_res)/(1+self.z),z_res) # D prop to a
        
        return P_m_psi
        
    def P_reion(self):

        P_patchy = 2 * self.Tb_mean()**2 * (self.bHI + self.mu**2 * self.f()) * self.P_m_psi()

        if self.verbose is True:
            print('At k = %f, mu = %f, z=%f, P_patchy = %f '% (self.k, self.mu, self.z, P_patchy))

        return P_patchy


class Fisher:

    def __init__(self,z,nk,nmu,dz, run_num, reion_scenario, exp_param, kmax = 0.4, verbose = True, wedgeon = True):

        self.z = z
        self.dz = dz
        self.kmax = kmax
        self.nk = nk
        self.nmu = nmu
        self.verbose = verbose
        self.wedgeon = wedgeon
        self.run_num = run_num
        self.reion_scenario = reion_scenario
        self.exp_name = exp_param['name']
        self.t_int = exp_param['t_int']
        self.bandwidth = exp_param['bandwidth']
        self.f_sky = exp_param['f_sky']
        self.dish_D = exp_param['D']
        

        self.params={}
        self.params['h'] = 0.6774
        # self.h = 0.6774
        self.h = self.params['h']
        self.params['Obh2'] = 0.02230
        self.params['Och2'] = 0.1188
        self.params['mnu'] = 0.194
        self.params['As'] = 2.142 # 10^9 * As
        self.params['ns'] = 0.9667
        self.params['alphas'] = -0.002
        self.params['taure'] = 0.066
        self.params['bHI'] = float(self.bHI(self.z))
        self.params['O_HI'] = self.OHI(self.z) * 1.e3
        self.OM = (self.params['Obh2']+self.params['Och2'])/self.h**2
        

        # zbin_width = self.DA(self.z + self.dz/2) - self.DA(self.z - self.dz/2)
        # self.kmin = 2 * np.pi / zbin_width

        self.F = None

        self.Finv = None
        self.Cov_inv_arr = None
        self.dparams_arr = None
        self.P_patchy_arr = None
        self.F_ss = None
        self.F_sg = None

        if (self.wedgeon is True):
            self.mus,self.dmu = self.mu_bins(self.mu_wedge(self.z), self.nmu)
        else:
            self.mus,self.dmu = self.mu_bins(0, self.nmu)
        self.zs = self.z_bins(3.5,5.5,self.dz)


    def OHI(self,z):
        if 3.49 < z < 4.5:
            return 1.18e-3
        elif 4.5 <= z < 5.51:
            return 0.98e-3

    # def bHI(self,z):
    #     bHI = np.array([2.2578, 2.3970, 2.5314, 2.6581, 2.7739])                           
    #     zs = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
    #     bHIs = interpolate.interp1d(zs,bHI)

    #     return bHIs(z)

    def bHI(self,z):
        # 1804.09180, Table 5
        if 3.49 < z < 4.5:
            bHI = 2.82
        elif 4.5 <= z <5.51:
            bHI = 3.18

        return bHI

    def DA(self,z):
        '''
        in units of Mpc
        '''
        zs = np.linspace(0,z,1000,endpoint=True)
        chi = 3.e5/(100*self.h)*integrate.simps(1./np.sqrt(1-self.OM+self.OM*(1+zs)**3),zs)

        return chi

    def V(self,z):
        '''
        Survey volume
        '''	
        A = 4 * np.pi * self.f_sky # in units of sr
        v = A / 3 * self.DA(z)**3

        return v

    def k_parall_wedge(self,z):
        '''
        output in units of 1/Mpc
        '''
        H_z = 100 * self.h * np.sqrt(1-self.OM+self.OM*(1+z)**3) # hubble constant in km/s/Mpc
        c = 3.e5 # km/s
        lambda_21 = 21.e-2 # 21cm wavelength in units of m
        k_parall = 2 * np.pi * self.dish_D * H_z / c / lambda_21 / (1+z)**2 

        return k_parall


    def k_perp_wedge(self,z):
        '''
        output in units of 1/Mpc
        '''
        lambda_21 = 21.e-2 # 21cm wavelength in units of m
        k_perp = 2*np.pi* self.dish_D/ lambda_21 / (1+z) / self.DA(z)

        return k_perp

    def k_parallel_min(self,z,dz):
        k_parallel_min = 2 * np.pi / (self.DA(z + dz/2) - self.DA(z - dz/2))
        return k_parallel_min

    def k_perp_min(self,z):
        k_perp_min = 2 * np.pi * self.dish_D / self.DA(z) / 0.21 / (1+z)
        return k_perp_min

    def mu_wedge(self,z):
        H_z = 100 * self.h * np.sqrt(1-self.OM+self.OM*(1+z)**3) # hubble constant in km/s/Mpc
        c = 3.e5 # in km/s
        mu = self.DA(z) * H_z / (c*(1+z)) / np.sqrt(1 + (self.DA(z) * H_z / (c * (1+z)))**2)

        return mu

    def k_bins(self,z, dz, nk):
        k_parallel_min = self.k_parallel_min(z,dz)
        k_perp_min = self.k_perp_min(z)

        kmin = np.sqrt(k_parallel_min**2+k_perp_min**2)
        ks0 = np.logspace(np.log10(kmin), np.log10(self.kmax), nk+1, endpoint = True)
        dks = np.diff(ks0)
        ks = 10**((np.log10(ks0[1:]) + np.log10(ks0[:-1])) / 2)

        return ks, dks, kmin

    def mu_bins(self, mu0, nmu):
        mus = np.linspace(mu0, 1, nmu+1, endpoint = True)

        mus = (mus[1:] + mus[:-1]) / 2
        
        dmu = (1-mu0) / nmu

        return mus, dmu

    def z_bins(self, zmin, zmax, dz):
        zs = np.linspace(zmin, zmax, int((zmax - zmin)/dz)+1, endpoint=True)
        zs = (zs[1:] + zs[:-1])/2

        return zs


    def dparams(self,k,mu,z):

        dP_dparam = []
        dparams={}
        dparams['h'] = 0.007
        dparams['Obh2'] = 0.0003
        dparams['Och2'] = 0.002
        dparams['mnu'] = 0.004
        dparams['As'] = 0.02 # 10^9 * dAs
        dparams['ns'] = 0.012
        dparams['alphas'] = 0.00002
        dparams['taure'] = 0.002

        dparams['bHI'] = 0.03
        dparams['O_HI'] = 0.01

        for i,par in enumerate(dparams.keys()):
            params_p = self.params.copy()
            params_m = self.params.copy()

            params_p['bHI'] = float(self.bHI(z))
            params_p['O_HI'] = self.OHI(z) * 1.e3
            params_m['bHI'] = float(self.bHI(z))
            params_m['O_HI'] = self.OHI(z) * 1.e3

            params_p[par] +=  dparams[par]
            params_m[par] -=  dparams[par]

            if par == 'bHI' or par == 'O_HI':
                P_fid_plus = P_21_obs(k,mu,z,params_p,'fid', self.t_int, self.bandwidth,self.exp_name).P_21obs()
                P_fid_minus = P_21_obs(k,mu,z,params_m,'fid', self.t_int, self.bandwidth,self.exp_name).P_21obs()

            else:
                P_fid_plus = P_21_obs(k,mu,z,params_p,par+'p', self.t_int, self.bandwidth,self.exp_name).P_21obs()
                P_fid_minus = P_21_obs(k,mu,z,params_m,par+'m', self.t_int, self.bandwidth,self.exp_name).P_21obs()

            # if par == 'As': 
            #     dparams[par] /= 1.e9

            # elif par == 'O_HI':
            #     dparams[par] /= 1.e3

            dP_dparam.append((P_fid_plus-P_fid_minus)/2/dparams[par])

        return dP_dparam

    def get_dparams(self):   
        dparams_arr = []
        for z in self.zs:
            this_arr = []
            ks, dks, kmin = self.k_bins(z,self.dz,self.nk)
            k_parallel_min = self.k_parallel_min(z,self.dz)
            k_perp_min = self.k_perp_min(z)
            if (self.wedgeon is True):
                mus, dmu = self.mu_bins(self.mu_wedge(z), self.nmu)
            else:
                mus,dmu = self.mu_bins(0, self.nmu)

            for k, dk in zip(ks,dks):
                # mu_min = kmin / k
                for mu in mus:
                    k_parallel = k * mu
                    k_perp = k * np.sqrt(1-mu**2)
                    if (k_parallel<k_parallel_min) or (k_perp<k_perp_min):
                        continue
                    # if (k_perp<k_perp_min): continue
                    # if (np.abs(mu) < k_parallel_min/k): 
                    #     continue
                    # if (np.abs(mu) > np.sqrt(1-(k_perp_min/k)**2)):
                    #     continue
                    # # exclude wedge k-space
                    if (self.wedgeon is True) and (np.abs(mu) < self.mu_wedge(z)):
                        continue                        
                    #     # k_parall = np.abs(k * mu)
                    #     # if (k_parall <= self.k_parall_wedge(z)) and (np.abs(mu) < self.mu_wedge(z)):
                    this_arr.append(self.dparams(k, mu, z))
            dparams_arr.append(this_arr)

        self.dparams_arr = np.array(dparams_arr)
        return self.dparams_arr


    def Cov_inv(self,k,mu,z, dk, dmu):

        nmodes = k**2 / (2 * np.pi)**2 * dk * dmu * (self.V(z+self.dz/2) - self.V(z-self.dz/2))
        this_param = self.params.copy()
        this_param['bHI'] = float(self.bHI(z))
        this_param['O_HI'] = self.OHI(z) * 1.e3

        P_fid = P_21_obs(k,mu,z, this_param,'fid', self.t_int, self.bandwidth,self.exp_name).P_21obs()

        return nmodes / 2. / P_fid**2

    def get_Cov_inv(self):
        dmu = self.dmu
        Cov_inv_arr = []

        for z in self.zs:
            this_arr = []
            ks, dks, kmin = self.k_bins(z,self.dz,self.nk)
            k_parallel_min = self.k_parallel_min(z,self.dz)
            k_perp_min = self.k_perp_min(z)
            if (self.wedgeon is True):
                mus, dmu = self.mu_bins(self.mu_wedge(z), self.nmu)
            else:
                mus,dmu = self.mu_bins(0, self.nmu)

            for k, dk in zip(ks,dks):
                # mu_min = kmin / k
                for mu in mus:
                    k_parallel = k * mu
                    k_perp = k * np.sqrt(1-mu**2)
                    if (k_parallel<k_parallel_min) or (k_perp<k_perp_min):
                        continue
                    # if (k_perp<k_perp_min): continue
                    # if (np.abs(mu) < k_parallel_min/k): 
                    #     continue
                    # if (np.abs(mu) > np.sqrt(1-(k_perp_min/k)**2)):
                    #     continue
                    # # exclude wedge k-space
                    if (self.wedgeon is True) and (np.abs(mu) < self.mu_wedge(z)):
                        continue                        
                        # k_parall = np.abs(k * mu)
                        # if (k_parall <= self.k_parall_wedge(z)) and (np.abs(mu) < self.mu_wedge(z)):
                    this_arr.append(self.Cov_inv(k, mu, z, dk, dmu))
            Cov_inv_arr.append(this_arr)

        self.Cov_inv_arr = np.array(Cov_inv_arr)
        return self.Cov_inv_arr

    def Fmat(self,k,mu,z, dk, dmu, Cov, dP_dparam):
        # zmax = 5.5
        # zmin = 3.5

        # P_reion = P_patchy_reion(k,mu,z,self.params).P_reion()

        F = np.zeros((len(self.params),len(self.params)))
        # dP_dparam = self.dparams(k,mu,z)

        # Cov = self.Cov_inv(k, mu, z, dk, dmu)

        for i in range(len(self.params)):
            for j in range(len(self.params)):
                F[i,j] = 2 * Cov * dP_dparam[i] * dP_dparam[j] # 2 times due to mu symmetry

        return F

    def get_F(self,z):
        print(z)
        # dmu = self.dmu

        if self.Cov_inv_arr is None:
            self.get_Cov_inv()

        Cov = self.Cov_inv_arr

        if self.dparams_arr is None:
            self.get_dparams()

        dparams = self.dparams_arr


        self.F = np.zeros((len(self.params),len(self.params)))
        z_index = int((z - 3.5)/0.2)

        k_parallel_min = self.k_parallel_min(z,self.dz)
        k_perp_min = self.k_perp_min(z)
        ks, dks, kmin = self.k_bins(z,self.dz,self.nk)

        if (self.wedgeon is True):
            mus, dmu = self.mu_bins(self.mu_wedge(z), self.nmu)
        else:
            mus, dmu = self.mu_bins(0, self.nmu)
        print(mus)

        count = 0
        for k, dk in zip(ks,dks):
            # mu_min = kmin / k
            for mu in mus:
                k_parallel = k * mu
                k_perp = k * np.sqrt(1-mu**2)
                if (k_parallel<k_parallel_min) or (k_perp<k_perp_min):
                    continue
                # if (k_perp<k_perp_min): continue
                # if (np.abs(mu) < k_parallel_min/k): 
                #     continue
                # if (np.abs(mu) > np.sqrt(1-(k_perp_min/k)**2)):
                #     continue
                # # exclude wedge k-space
                if (self.wedgeon is True) and (np.abs(mu) < self.mu_wedge(z)):
                    continue                        
                # if self.verbose is True:
                print('k= %f, mu = %f' % (k, mu))      
                print('z_index = %d, count=%d' % (z_index, count))          
                self.F += self.Fmat(k,mu,z, dk, dmu, Cov[z_index][count], dparams[z_index][count]) 
                count += 1

        return self.F



    def param_mean(self, param_num, data):
    
        return np.sum(data.T[0]*data.T[param_num])/np.sum(data.T[0])

    def param_cov(self, param_nums, data):

        param1 = param_nums[0]
        param2 = param_nums[1]
        param1_mean = self.param_mean(param1, data)
        param2_mean = self.param_mean(param2, data)
        
        return np.sum(data.T[0]*(data.T[param1]-param1_mean)*(data.T[param2]-param2_mean))/np.sum(data.T[0])


    def CMB_cov(self, file_name):

        data = np.loadtxt(file_name)

        param_planck = {}
        param_planck['Obh2'] = 2
        param_planck['Och2'] = 3
        param_planck['taure'] = 5
        param_planck['mnu'] = 6
        param_planck['ns'] = 8
        param_planck['H0'] = 30
        param_planck['10^9As'] = 44

        cov_planck = np.zeros((7,7))
        for i,par1 in enumerate(param_planck.keys()):
                for j,par2 in enumerate(param_planck.keys()):
                    param1 = param_planck[par1]
                    param2 = param_planck[par2]
                    cov_planck[i,j] += self.param_cov([param1,param2],data)
                #     if j==5: cov_planck[i,j] /= 100
                #     # if j==5: cov_planck[i,j] /= 1.e9
                # if i==5: cov_planck[i,j] /= 100
                # # if i==5: cov_planck[i,j] /= 1.e9
        cov_planck[5,:] /= 100
        cov_planck[:,5] /= 100

        return cov_planck

    def get_F_prior(self, cov_prior):
        '''
        Add CMB priors to 21cm Fisher matrix
        '''

        # 21cm Fisher matrix parameter index mapping
        param_map = {}
        param_map['Obh2'] = 1
        param_map['Och2'] = 2
        param_map['taure'] = 7
        param_map['mnu'] = 3
        param_map['ns'] = 5
        param_map['H0'] = 0
        param_map['As'] = 4

        F_CMB = np.linalg.inv(cov_prior)
        F_prior = np.zeros((8,8))

        for i,par1 in enumerate(param_map.keys()):
            for j,par2 in enumerate(param_map.keys()):
                pos1 = param_map[par1]
                pos2 = param_map[par2]
                F_prior[pos1,pos2] += F_CMB[i,j]

        return F_prior

    def get_Finv(self,z):
        self.cov_prior=np.zeros((7,7))
        if self.F is None:
            print('Generate Fishser matrix\n')
            self.F = self.get_F(z)

        else:
            for i in range(4):
                self.cov_prior += self.CMB_cov('./base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_%d.txt' % (i+1))
            
            F_prior = self.add_prior(self.cov_prior)
            self.Finv = np.linalg.inv(F_prior)
        
        return self.Finv


    def get_P_patchy(self):
        P_patchy_arr = []

        params = self.params.copy()


        for z in self.zs:
            params['bHI'] = float(self.bHI(z))
            params['O_HI'] = self.OHI(z) * 1.e3

            z_index = int((z - 3.5)/0.2)

            k_parallel_min = self.k_parallel_min(z,self.dz)
            k_perp_min = self.k_perp_min(z)
            ks, dks, kmin = self.k_bins(z, self.dz, self.nk)

            count = 0
            this_P_patchy = []
            if (self.wedgeon is True):
                mus, dmu = self.mu_bins(self.mu_wedge(z), self.nmu)
            else:
                mus,dmu = self.mu_bins(0, self.nmu)

            for k, dk in zip(ks,dks):
                # mu_min = kmin / k
                for mu in mus:
                    k_parallel = k * mu
                    k_perp = k * np.sqrt(1-mu**2)
                    if (k_parallel<k_parallel_min) or (k_perp<k_perp_min):
                        continue
                    # if (k_perp<k_perp_min): continue
                    # if (np.abs(mu) < k_parallel_min/k): 
                    #     continue
                    # if (np.abs(mu) > np.sqrt(1-(k_perp_min/k)**2)):
                    #     continue
                    # # exclude wedge k-space
                    if (self.wedgeon is True) and (np.abs(mu) < self.mu_wedge(z)):
                        continue                        
                        # k_parall = np.abs(k * mu)
                        # if (k_parall <= self.k_parall_wedge(z)) and (np.abs(mu) < self.mu_wedge(z)):
                    P_reion = P_patchy_reion(k,mu,z,params, './21cmFAST/catalinas_codes/21cm_xh_%s_%s.txt' % (self.run_num, self.reion_scenario),'./21cmFAST/matter_cross_HI_%s_%s.txt' % (self.run_num, self.reion_scenario)).P_reion()
                    this_P_patchy.append(P_reion)
                    count += 1
            P_patchy_arr.append(this_P_patchy)

        self.P_patchy_arr = P_patchy_arr
        return self.P_patchy_arr


    def this_param_shift(self,z, k,mu, dk, dmu, param, Cov, dP_dparam, P_reion):
        # if 3.5 <= z <= 4.5:
        #     F_sg = self.F_sg[0]
        #     F_ss = self.F_ss[0]
        # elif 4.5 < z <= 5.5:
        #     F_sg = self.F_sg[1]
        #     F_ss = self.F_ss[1]

        params = self.params.copy()
        d_param = 0

        # self.dP_dparam = self.dparams(k,mu,z)

        if self.Finv is None:
            raise Exception("Need an Finv to continue")
            # self.get_Finv()

        gen_param_num = 8


        # if param_index < gen_param_num: 
        for j in range(len(params)):
            param_index = list(params.keys()).index(param)
            if 3.5 <= z <= 4.5:
                d_param += self.Finv[param_index,j] * Cov * P_reion * dP_dparam[j]
            elif 4.5 < z <= 5.5:
                # for b_HI and O_HI , there are two 2X2 blocks for different z bins, the second bin has param_index (10, 11)
                if param_index >= gen_param_num: 
                    param_index += 2

                if j < gen_param_num:
                    d_param += self.Finv[param_index,j] * Cov * P_reion * dP_dparam[j]
                elif gen_param_num <= j <= (gen_param_num + 1):
                    d_param += self.Finv[param_index,j+2] * Cov * P_reion * dP_dparam[j]

        # for j in range(8):
        #     d_param += self.Finv[param_index,j] * Cov * P_reion * dP_dparam[j]
        #     for i in range(2):
        #         d_param -= self.Finv[param_index,j] * Cov * P_reion * (F_sg.T @ np.linalg.inv(F_ss))[j,i] * dP_dparam[i+8] 

        return d_param

    def param_shift(self, param):
        shift = []
        if self.P_patchy_arr is None:
            self.get_P_patchy()

        for z in self.zs:

            k_parallel_min = self.k_parallel_min(z,self.dz)
            k_perp_min = self.k_perp_min(z)
            ks, dks, kmin = self.k_bins(z, self.dz, self.nk)

            if (self.wedgeon is True):
                mus, dmu = self.mu_bins(self.mu_wedge(z), self.nmu)
            else:
                mus,dmu = self.mu_bins(0, self.nmu)

            z_index = int((z - 3.5)/0.2)
            Cov = self.Cov_inv_arr
            dP_dparam = self.dparams_arr
            P_reion = self.P_patchy_arr
            count = 0
            this_shift = []
            for k, dk in zip(ks,dks):
                # mu_min = kmin / k
                for mu in mus:
                    k_parallel = k * mu
                    k_perp = k * np.sqrt(1-mu**2)
                    if (k_parallel<k_parallel_min) or (k_perp<k_perp_min):
                        continue
                    # if (k_perp<k_perp_min): continue
                    # if (np.abs(mu) < k_parallel_min/k): 
                    #     continue
                    # if (np.abs(mu) > np.sqrt(1-(k_perp_min/k)**2)):
                    #     continue
                    # # exclude wedge k-space
                    if (self.wedgeon is True) and (np.abs(mu) < self.mu_wedge(z)):
                        continue                        
                        # if (k_parall <= self.k_parall_wedge(z)) and (np.abs(mu) < self.mu_wedge(z)):
                    # times 2 because only picked positive mus
                    this_shift.append(2*self.this_param_shift(z, k, mu, dk, dmu, param, Cov[z_index][count], dP_dparam[z_index][count], P_reion[z_index][count]))
                    count += 1
            shift.append(this_shift)

        return shift

class get_shifts:
    def __init__(self, exp_param, run_num, reion_scenario, wedgeon):
        self.z = 3.5
        self.exp_param = exp_param
        self.run_num = run_num
        self.reion_scenario = reion_scenario
        self.wedgeon = wedgeon

        self.t_int = exp_param['t_int']
        self.bandwidth = exp_param['bandwidth']
        self.exp_name = exp_param['name']

        self.nk = 30
        self.nmu = 10
        self.dz = 0.2
        self.Fisher = Fisher(self.z, self.nk, self.nmu, self.dz, self.run_num, self.reion_scenario, self.exp_param, verbose=False, wedgeon=wedgeon)

        # if self.dparams_arr is None:
        #     self.get_dP_dparams()
        # if self.Cov_inv_arr is None:
        #     self.get_Nmode_by_sigma2()
        # if self.P_patchy_arr is None:
        #     self.get_P_patchy()

    def get_dP_dparams(self):
        self.dparams_arr = self.Fisher.get_dparams()
        
        return self.dparams_arr

    def get_Nmode_by_sigma2(self):
        self.Cov_inv_arr = self.Fisher.get_Cov_inv()

        return self.Cov_inv_arr

    def get_P_patchy(self):
        self.P_patchy_arr = self.Fisher.get_P_patchy()

        return self.P_patchy_arr

    def get_fid_Fisher(self):
        Fs = []
        zs = np.arange(3.6,5.5,0.2)
        for z in zs:
            print('z=%f' % z)
            this_Fisher = Fisher(z, self.nk, self.nmu, self.dz, self.run_num, self.reion_scenario, self.exp_param, verbose=False, wedgeon=self.wedgeon)
            this_Fisher.Cov_inv_arr = self.Cov_inv_arr
            this_Fisher.dparams_arr = self.dparams_arr 
            thisF = this_Fisher.get_F(z)
            Fs.append(thisF)

        F_1 = Fs[0]+Fs[1]+Fs[2]+Fs[3]+Fs[4]
        F_2 = Fs[5]+Fs[6]+Fs[7]+Fs[8]+Fs[9]

        Fgg_1 = F_1[:8,:8] # 8 x 8
        Fsg_1 = F_1[8:,:8] # 2 x 8
        Fss_1 = F_1[8:,8:] # 2 x 2

        Fgg_2 = F_2[:8,:8] # 8 x 8
        Fsg_2 = F_2[8:,:8] # 2 x 8
        Fss_2 = F_2[8:,8:] # 2 x 2

        F_tot= np.block([\
            [Fgg_1 + Fgg_2, Fsg_1.T, Fsg_2.T],\
            [Fsg_1, Fss_1, np.zeros((2,2))],\
            [Fsg_2, np.zeros((2,2)), Fss_2]\
            ])

        return F_tot

    def get_F_planck_prior(self):
        cov_planck = np.zeros((7,7))
        for i in range(4):
            cov_planck += self.Fisher.CMB_cov('./base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_%d.txt' %(i+1))
        F_prior = np.zeros((12,12))
        F_prior[:8,:8] += self.Fisher.get_F_prior(cov_planck)

        return F_prior

    def get_Finv(self,extra_F_prior=np.zeros((12,12))):
        F_tot = self.get_fid_Fisher() + self.get_F_planck_prior() + extra_F_prior

        Finv = np.linalg.inv(F_tot)
        return Finv

    def shifts(self, extra_F_prior=np.zeros((12,12)), print_out=True):
        self.Fisher.Finv = self.get_Finv(extra_F_prior)
        self.Fisher.Cov_inv_arr = self.Cov_inv_arr
        self.Fisher.dparams_arr = self.dparams_arr 
        self.Fisher.P_patchy_arr = self.P_patchy_arr

        shifts = {}
        for i,par in enumerate(self.Fisher.params):
            shifts[par] = self.Fisher.param_shift(par)

        if print_out:
            param_shift = {}
            for i,par in enumerate(self.Fisher.params):
                if i<8:
                    param_shift[par] = np.sum(np.sum(shifts[par]))
            print(param_shift)
        return shifts






############ script example ############

# run_num_list = ['r1', 'r2', 'r3', 'r4']
# reion_scenario = ['early', 'late', 'planck']
# t_int = [100, 1000]
# bandwidth = {'SKA': 300, 'PUMA': 200}


# Forecast = patchy_reion_ns.Fisher(3.5, 30, 10, 0.2, run_num_list[0], reion_scenario[0], t_int[0], bandwidth['PUMA'], verbose=False)
# dparams_arr = Forecast.get_dparams()
# Cov_inv_arr = Forecast.get_Cov_inv()
# P_patchy_arr = Forecast.get_P_patchy()

# # get whole Fs
# Fs = []
# zs = np.arange(3.6,5.5,0.2)
# for z in zs:
#     Forecast = patchy_reion_ns.Fisher(z, 30, 10, 0.2, run_num_list[0], reion_scenario[0], t_int[0], bandwidth['PUMA'], verbose=False)
#     Forecast.Cov_inv_arr = Cov_inv_arr
#     Forecast.dparams_arr = dparams_arr 
#     thisF = Forecast.get_F()
#     Fs.append(thisF)

# F_1 = Fs[0]+Fs[1]+Fs[2]+Fs[3]+Fs[4]
# F_2 = Fs[5]+Fs[6]+Fs[7]+Fs[8]+Fs[9]

# Fgg_1 = F_1[:8,:8] # 8 x 8
# Fsg_1 = F_1[8:,:8] # 2 x 8
# Fss_1 = F_1[8:,8:] # 2 x 2

# Fgg_2 = F_2[:8,:8] # 8 x 8
# Fsg_2 = F_2[8:,:8] # 2 x 8
# Fss_2 = F_2[8:,8:] # 2 x 2

# cov_planck = Forecast.CMB_cov('/Users/heyang/Downloads/COM_CosmoParams_fullGrid_R3/base_mnu/plikHM_TTTEEE_lowl_lowE_lensing/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_2.txt')
# F_prior = Forecast.get_F_prior(cov_planck)

# F_tot_prior = np.block([
#     [Fgg_1 + Fgg_2 + F_prior, Fsg_1.T, Fsg_2.T],
#     [Fsg_1, Fss_1, np.zeros((2,2))],
#     [Fsg_2, np.zeros((2,2)), Fss_2]
# ])

# Finv = np.linalg.inv(F_tot_prior)

# Forecast.Finv = Finv

# shifts = {}
# for i,par in enumerate(Forecast.params):
#     shifts[par] = Forecast.param_shift(par)




