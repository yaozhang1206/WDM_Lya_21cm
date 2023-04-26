import numpy as np
from scipy.integrate import quad


class lya_convert(object):
    # this class defines the convertor for our fiducial cosmology, remember that this should not depend on cosmology since one usually ignore the change of the covariance with cosmology (see the verde & bernal paper (2020) on forecasts errors
    
    def __init__(self):
        # need to define cosmology for hubble parameter
        self.Omega_m = 0.3088
        self.h = 0.6774
        self.Omega_r = 8.6e-5 # for consistency with 21cmFAST, but pretty much useless here
#        print 'Paulo it might be a good idea to use omegas instead of Omegas\n'
        
#        print 'Paulo check the referee report conversion factor because I might have forgotten the extra redshift factor\n' # was checked no problem. 
    

    def hubble(self,z):
        # returns the hubble parameter as a function of z for given cosmology in the constructor
        return (self.h*100)*np.sqrt(self.Omega_m*(1 + z)**3 + self.Omega_r*(1 + z)**4 + (1.0 - self.Omega_m))

    
    
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
        # self explanatory haha
        # NOTE THAT THIS IS SUPPOSED TO BE angular_diameter_distance(z) * (1+z), and the angular diameter distance is D_A = D_m / (1+z), with D_m the comoving distance, hence it should be just D_m
        temp = quad(self.integrand, 0, z)
#        return temp[0] / (1.0 + z) # typo mentioned in the note above
        return temp[0]
    
    def dMpc_ddeg(self,z):
        # this one takes care of the transition from deg to Mpc, so twice for areas!!
        # how to relate Mpc with radian? that's how D_A, the comoving angular diameter distance is defined!
        dMpc_drad = self.comoving_dist(z) # / (1.0 + z) # activate this if I think Andreu was wrong
        # it seems that what we needed is not the angular diameter distance but the comoving distance?
        """OK AFTER REDOING THE WHOLE CODE I THINK MY CONCLUSION IS THAT WHAT ANDREU PUT WAS THE COMOVING DISTANCE THAT MPC TRANSLATE TO RADIANS, THIS IS A LITTLE BIT WEIRD SO FLAG FOR THE FUTURE"""
        
#        print('Comoving angular diameter distance = ',dMpc_drad)
        return dMpc_drad * np.pi / 180.0
