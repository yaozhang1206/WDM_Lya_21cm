import numpy as np
from scipy.integrate import quad

"""
    This class defines the unit convertor using our fiducial cosmology.
"""

class lya_convert(object):

    def __init__(self):
        # need to define cosmology for hubble parameter
        self.Omega_m = 0.3088
        self.h = 0.6774
        self.Omega_r = 8.6e-5 # for consistency with 21cmFAST, but pretty much useless here   

    def hubble(self,z):
        # returns the hubble parameter as a function of z for given cosmology in the constructor
        return (self.h*100)*np.sqrt(self.Omega_m*(1 + z)**3 + self.Omega_r*(1 + z)**4 + (1.0 - self.Omega_m))

    def dMpc_dz(self,z):
        # relate Mpc to redshift through integrand of comoving distance
        c_kms = 2.99979e5
        return c_kms / self.hubble(z)
    
    def dkms_dMpc(self,z):
        # note that I include my little h here
        # returns the factor needed to change from Mpc to km/s, i.e. multiply by H(z)/(1 + z)
        return self.hubble(z)/(1.0 + z)
        
    def dkms_dlobs(self,z):
        # to convert from lambda to km/s
        c_kms = 2.99979e5
        return c_kms / 1215.67 / (1.0 + z)
        
    def dMpc_dlobs(self,z):
        # conversion factor from lambda to Mpc
        return self.dkms_dlobs(z) / (1.0 * self.dkms_dMpc(z))
    
    def integrand(self,z):
        c_kms = 2.99979e5
        return 1.0 * c_kms / (1.0 * self.hubble(z))
    
    def comoving_dist(self,z):
        # NOTE THAT THIS IS SUPPOSED TO BE angular_diameter_distance(z) * (1+z), and the angular diameter distance is D_A = D_m / (1+z), with D_m the comoving distance, hence it should be just D_m
        temp = quad(self.integrand, 0, z)
        return temp[0]
    
    def dMpc_ddeg(self,z):
        # this one takes care of the transition from deg to Mpc, so twice for areas!!
        # What we needed is not the angular diameter distance but the comoving distance because we are not calculating proper distance but the comoving distance.
        dMpc_drad = self.comoving_dist(z)
        return dMpc_drad * np.pi / 180.0
