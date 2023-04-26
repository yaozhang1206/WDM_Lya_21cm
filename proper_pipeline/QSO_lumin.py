import numpy as np
import scipy.interpolate
#import QLF_FDR_vrs_SV as QLFS

class QSO_LF(object):
    # This constructs the luminosity function required for the forecast
    
    def __init__(self,QLF_verbose):
        self.QLF_verbose = QLF_verbose
        if (self.QLF_verbose == 1):
            print('Using the SV QLF')
        else:
            print('Using the default QLF')
#        self.my_QLFS = QLFS.QLFS()
        # Construct our luminosity function
        self._setup_LF()
    
    def _setup_LF(self):
        # Grab the DESI data
        file = './data/dNdzdg_QSO.dat'
        z,m,temp_dNdmdzddeg2 = np.loadtxt(file,unpack=True)
        # now grab the arrays for later interpolation
        z = np.unique(z)
        m = np.unique(m)
#        print 'Paulo says:\n'
#        print len(z), z
#        print len(m), m
#        print len(temp_dNdmdzddeg2), temp_dNdmdzddeg2
        # Paulo needs some help with this arrays so here you/I go
        self.z_array = z
        self.m_array = m
        
        # the bin from data seem to imply a dz = 0.2 and dm = 0.5
        dz = 0.2
        dm = 0.5
        # compute the derivative
        temp_dNdmdzddeg2 /= (1.0 * dz * dm)
        temp_dNdmdzddeg2 = np.reshape(temp_dNdmdzddeg2,[len(z),len(m)])
        
#        print len(temp_dNdmdzddeg2), temp_dNdmdzddeg2
        
        # compute the range (remember the center a bin spans z - delta_z/2 to z + delta_z/2
        self.zmin = z[0] - 0.5 * dz
        self.zmax = z[-1] + 0.5 * dz
        
#        print self.zmin, self.zmax
        # time for the magnitude
        self.mmin = m[0] - 0.5*dm
        self.mmax = m[-1] + 0.5*dm
        
#        print self.mmin, self.mmax
        # interpolate for eveyother redshift/magnitude
        self._dNdmdzddeg2 = scipy.interpolate.RectBivariateSpline(z,m,temp_dNdmdzddeg2, bbox=[self.zmin,self.zmax,self.mmin,self.mmax], kx=2,ky=2)
    
    def range_z(self):
        # this function computes the range of redshifts for QSO in lum. func.
        return self.zmin,self.zmax
    
    def range_mag(self):
        # same as prior but for the magnitude
        return self.mmin,self.mmax
        
    def dNdzdmddeg2(self,z_q,m_q):
        # Luminosity function of quasars, notice that the units have the per redshift, per observed magnitude and square degrees
        if (self.QLF_verbose == 1):
            
            return self._dNdmdzddeg2(z_q,m_q) * self.QLF_ratios(z_q)
        else:
            return self._dNdmdzddeg2(z_q,m_q)

    def QLF_arrays(self):
        # this function returns the redshifts and magnitudes of the data file
        return self.z_array,self.m_array

    def QLF_ratios(self,z_q):
        # The comparison is done with the "plotted" luminosity functions
        # the deltas of the bins
        dm = 0.5
        dz = 0.2
        # the SV luminosity function
        dn_dz_SV = self.my_QLFS.SV_dn_dz(z_q) / dm / dz # get rid-off dz if doing plotted
        # now for the default one
        dn_dz_default = np.sum(self._dNdmdzddeg2(z_q,self.m_array)) * dm  # * dz if doing plotted
        # the ratio
        ratio = dn_dz_SV / dn_dz_default
        # test
#        print('SV ',dn_dz_SV)
#        print('Default',dn_dz_default)
#        print('ratio ',ratio)
        return ratio
