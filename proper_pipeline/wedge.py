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

class bins:
    def __init__(self, dish_D, verbose = True, wedgeon = True):

        self.z = 3.5
        self.dz = 0.2
        self.kmax = 0.4
        self.nk = 30
        self.nmu = 5
        self.verbose = verbose
        self.wedgeon = wedgeon
        self.dish_D = dish_D
        
        self.h = 0.6774
        self.OM = (0.02230+0.1188)/self.h**2


    def DA(self, z):
        '''
        in units of Mpc
        '''
        zs = np.linspace(0,z,1000,endpoint=True)
        chi = 3.e5/(100*self.h)*integrate.simps(1./np.sqrt(1-self.OM+self.OM*(1+zs)**3),zs)

        return chi
    
    def k_parallel_min(self):
        k_parallel_min = 2 * np.pi / (self.DA(self.z + self.dz/2) - self.DA(self.z - self.dz/2))
        return k_parallel_min
    
    def k_perp_min(self):
        k_perp_min = 2 * np.pi * self.dish_D / self.DA(self.z) / 0.21 / (1+self.z)
        return k_perp_min
    
    def mu_wedge(self):
        H_z = 100 * self.h * np.sqrt(1-self.OM+self.OM*(1+self.z)**3) # hubble constant in km/s/Mpc
        c = 3.e5 # in km/s
        mu = self.DA(self.z) * H_z / (c*(1+self.z)) / np.sqrt(1 + (self.DA(self.z) * H_z / (c * (1+self.z)))**2)
        return mu

    def k_bins(self):
        k_parallel_min = self.k_parallel_min(self)
        k_perp_min = self.k_perp_min(self)

        kmin = np.sqrt(k_parallel_min**2+k_perp_min**2)
        ks0 = np.logspace(np.log10(kmin), np.log10(self.kmax), self.nk+1, endpoint = True)
        dks = np.diff(ks0)
        ks = 10**((np.log10(ks0[1:]) + np.log10(ks0[:-1])) / 2)

        return ks, dks, kmin

    def mu_bins(self):
        mu_wedge = self.mu_wedge(self)
        mus = np.linspace(mu_wedge, 1, self.nmu+1, endpoint = True)

        mus = (mus[1:] + mus[:-1]) / 2
        
        dmu = (1- mu_wedge) / self.nmu

        return mus, dmu