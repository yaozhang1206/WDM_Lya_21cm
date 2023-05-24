import numpy as np
from scipy import integrate

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
        k_parallel_min = self.k_parallel_min()
        k_perp_min = self.k_perp_min()

        kmin = np.sqrt(k_parallel_min**2+k_perp_min**2)
        ks0 = np.logspace(np.log10(kmin), np.log10(self.kmax), self.nk+1, endpoint = True)
        dks = np.diff(ks0)
        ks = 10**((np.log10(ks0[1:]) + np.log10(ks0[:-1])) / 2)

        return ks, dks, kmin

    def mu_bins(self):
        mu_wedge = self.mu_wedge()
        mus = np.linspace(mu_wedge, 1, self.nmu+1, endpoint = True)

        mus = (mus[1:] + mus[:-1]) / 2
        
        dmu = (1- mu_wedge) / self.nmu

        return mus, dmu
 
'''
ska = bins(dish_D=40.)
puma = bins(dish_D=6.)

zobs = [5.5,5.0,4.5,4.0,3.5,3.0]

for z in zobs:
    ska.z = z
    puma.z = z
    print("z = %.1f"%z)
    print("wedge: %f"%ska.mu_wedge())
    print("ska:  kmin: %f  k_parallel_min: %f  k_perp_min: %f"%(np.sqrt(ska.k_parallel_min()**2+ska.k_perp_min()**2), ska.k_parallel_min(), ska.k_perp_min()))
    print("puma: kmin: %f  k_parallel_min: %f  k_perp_min: %f"%(np.sqrt(puma.k_parallel_min()**2+puma.k_perp_min()**2), puma.k_parallel_min(), puma.k_perp_min()))
    
z = 5.5
wedge: 0.935066
ska:  kmin: 0.069163  k_parallel_min: 0.065444  k_perp_min: 0.022373
puma: kmin: 0.065530  k_parallel_min: 0.065444  k_perp_min: 0.003356
z = 5.0
wedge: 0.926315
ska:  kmin: 0.063257  k_parallel_min: 0.058103  k_perp_min: 0.025011
puma: kmin: 0.058224  k_parallel_min: 0.058103  k_perp_min: 0.003752
z = 4.5
wedge: 0.915296
ska:  kmin: 0.058391  k_parallel_min: 0.051070  k_perp_min: 0.028308
puma: kmin: 0.051247  k_parallel_min: 0.051070  k_perp_min: 0.004246
z = 4.0
wedge: 0.901112
ska:  kmin: 0.055015  k_parallel_min: 0.044363  k_perp_min: 0.032535
puma: kmin: 0.044631  k_parallel_min: 0.044363  k_perp_min: 0.004880
z = 3.5
wedge: 0.882384
ska:  kmin: 0.053834  k_parallel_min: 0.038000  k_perp_min: 0.038132
puma: kmin: 0.038428  k_parallel_min: 0.038000  k_perp_min: 0.005720
z = 3.0
wedge: 0.856906
ska:  kmin: 0.055923  k_parallel_min: 0.032006  k_perp_min: 0.045858
puma: kmin: 0.032737  k_parallel_min: 0.032006  k_perp_min: 0.006879

'''
