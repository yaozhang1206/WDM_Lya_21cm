# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate

"""
    This calculates the filtering mass M_F and filetering scale k_F in Equations (11), (12), (13), which quantifies the smoothing of small-scale baryonic structure and the reduction of gas in low-mass halos due to the increased pressure after rreionization.
    The input data is the thermal history (sound speed) extracted from Gadget-2 simulations. 
"""

def load_f(name):
	with open(name,'r') as f:
		data=f.read()
	return data


class f_scale:
	def __init__(self,file_name,a, z_re):
		self.z_re = z_re
		parsec=3.085677581e16 # m per parsec
		self.file_name=file_name	
		self.H_0=67.74e3/(parsec*10**6) # in units of s^-1, Hubble constants now, 67.74 km/s/mpc
		self.Omega_m=0.3089 # Omega_m = 0.3089+-0.0062
		self.G=6.674e-11  #6.674*10^11 m3*kg−1*s−2 ### 4.30091(25)*10−3 pc*M_solar-1*(km/s)^2
		self.solar_m= 1.98847e30 #(1.98847±0.00007)*10^30 kg
		self.a_dec=0.000942791416826941
		self.a_t=a # final scale factor
		self.t_dec=2*self.a_dec**(3./2)/3/self.H_0/np.sqrt(self.Omega_m)
		self.t= 2*self.a_t**(3./2)/3/self.H_0/np.sqrt(self.Omega_m)
		self.psi_dec=self.psi(self.t_dec)
		self.rho_mo=3*self.Omega_m*self.H_0**2/8/np.pi/self.G 

	def cs_extract(self): 
		# extract sound speed from Gadget-2 snapshots
		k=1.38069e-23 # Boltzmann constant @ J*K-1
		m_p=1.67262192369e-27 # proton mass @ kg
		data=load_f(self.file_name)
		data=[float(i) for i in data.split()]
		data=np.reshape(np.array(data),(int(len(data)/9),9))
		T_0=np.mean(data[:,4][np.abs(data[:,3]-1)<0.01])
		logT_0=np.mean(np.log10(data[:,4][np.abs(data[:,3]-1)<0.01]))
		rho_mean = np.mean(data[:,3])
		gamma=np.array([((np.log10(i[4])-np.log10(T_0))/np.log10(i[3])+1) for i in data if (i[3]-1>0.05 and i[3]<300)])
		gamma=np.mean(gamma[:][gamma>-1]) 
		mu=0.6127*m_p # reduced mass @ H+/He+/e- plasma with 24.53% He+
		c_s = np.sqrt(gamma*k*T_0/mu) # calculate sound speed
		return c_s, gamma, T_0, logT_0
	def kernel(self,psi):
		return 2*(1-3*self.psi_dec/psi+2*(self.psi_dec/psi)**(3./2))*(1-psi**(1./2))/(1-3*self.psi_dec+2*self.psi_dec**(3./2))
	def psi(self,time):
		return self.a(time)/self.a_t
	def a(self,time):
		return (3*self.H_0*np.sqrt(self.Omega_m)/2)**(2./3)*time**(2./3)
	def a_deriv(self,time):
		return 2./3.*(3*np.sqrt(self.Omega_m)*self.H_0/2)**(2./3)*time**(-1./3)
	def results(self,c_s,v_bc):
		psi=self.psi_dec
		t=self.t_dec
		kF_s_inverse2=0
		dpsi=(1-self.psi_dec)/1000.
		for i in range(1000):
			dt=3./2*np.sqrt(psi)*self.t*dpsi
			t+=dt
			psi+=dpsi
			if self.a(t)<1./(1+float(self.z_re)): 
				continue

			kF_s_inverse2+=self.kernel(psi)*c_s(self.a(t)+1.e-8)**2/(self.a_deriv(t))**2*dpsi
		kF_v_inverse2=3*(v_bc*self.t_dec/self.a_dec)**2*((-3./2.*self.psi_dec*np.log(self.psi_dec)+3*self.psi_dec**(3./2)-3*self.psi_dec //
			-9./2*self.psi_dec*(1-self.psi_dec**(1./2))**2*(1+2*self.psi_dec**(1./2))**(-1))/(1-3*self.psi_dec+2*(self.psi_dec)**(3./2)))
		kF_inverse2=kF_v_inverse2+kF_s_inverse2
		kF=np.sqrt(1/kF_inverse2)
		MF=self.rho_mo*4./3*np.pi**4*(kF_inverse2)**(3./2)/self.solar_m*67.74/100 # in units of M_solar/h
		return MF,kF_v_inverse2,kF_s_inverse2,kF,kF_inverse2


