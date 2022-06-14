import numpy as np
import scipy.interpolate

class QSO_LF(object):
    # This constructs the luminosity function required for the forecast, i.e. the QLF of how many quasars will DESI see as a function of redshift of observation and depending on their magnitued (how dim/bright they are)
    
    def __init__(self):
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
        return self._dNdmdzddeg2(z_q,m_q)

