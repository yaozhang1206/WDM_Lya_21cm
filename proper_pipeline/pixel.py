import numpy as np
from scipy.interpolate import RegularGridInterpolator

# this one is for the pixel error
class spec_pixel(object):
    # This class describe given spectrograph, and gives noise estimates

    def __init__(self, band):
        # initialize this constructor
        self.band = band
        if not self._setup():
            print("couldn't setup the constructor")
            raise SystemExit
    
    def _read_file(self,mag):
        # this get the files with SNR(z,lambda)
        if self.band is 'g':
            file = '../data/sn-spec-lya-g'+str(mag)+'-t4000.dat'
        elif self.band is 'r':
#            print 'Made it to r-band\n'
            file = '../data/sn-spec-lya-r'+str(mag)+'-t4000.dat'
        else:
            print('specify band please', self.band)
            raise SystemExit
        data = np.loadtxt(file)
        l_A = data[:,0]
        SN = data[:,1:]
        return l_A,SN
        
    def _setup(self):
        # Files grab from the andreu repository, supposed to come from desihub/desimodel/bin/desi_quicklya.py
        # number of exposures in file with SNR
        self.file_Nexp = 4
        # What about redshifts and magnitudes?
        self.mags = np.arange(19.25,25.0,0.50)
        self.zq = np.arange(2.0,4.9,0.25)
        
        # pixel wavelengths in file
        self.lobs_A = None
        # signal to noise per pixel in file
        self.SN = None
        i = 0
        for i in range(len(self.mags)):
            m = self.mags[i]
            l_A,SN = self._read_file(m)
            if i == 0:
                self.lobs_A = l_A
                self.SN = np.empty((len(self.mags),len(self.zq),len(self.lobs_A)))
            self.SN[i,:,:] = SN.transpose()
        
        # now need to be able to cover the range randomly, so interpolation time
        self.SN = RegularGridInterpolator((self.mags,self.zq,self.lobs_A),self.SN)
        return True
        
    # now the typical functions to figure out the range of parameters
    # start with redshift
    def range_zq(self):
        return self.zq[0], self.zq[-1]
    
    # then magnitudes
    def range_mag(self):
        return self.mags[0], self.mags[-1]
    
    # finally the wavelength range
    def range_lobs_A(self):
        return self.lobs_A[0], self.lobs_A[-1]
        
    # time to get the pixel noise needed
    def PixelNoiseRMS(self,rmag,zq,lobs_A,pix_A,Nexp=4):
        # Normalized RMS noise as a function of observed magnitude, redshift of the quasar, pixel wavelength (amstrongs), and piwel width (also in armstrongs). This is normalized in the sense that this is the noise for delta_flux, not flux, and brigther quasars will have less normalized noise. According to Andreu this is the inverse of signal to noise. If SN = 0, or not covered, return very large number.
        large_noise = 1.0e10
        # if the observed magnitude is just way too high, then throw a lot of noise
        if rmag > self.mags[-1]: return large_noise
        # but what to do if brightness is smaller than minimum m?
        
        # Andreu went for use the minimum, otherwise one can extrapolate
        # Let's use our minimum since it seems easier
        temp_rmag = np.fmax(rmag, self.mags[0])
        
        
        # if quasar is not in the observed range of redshifts, then it must be noisy
        if zq > self.zq[-1] or zq < self.zq[0]: return large_noise
        # finally if wavelength also is out of range
        if lobs_A > self.lobs_A[-1] or lobs_A < self.lobs_A[0]: return large_noise
    

        # the file has SN per Angstrom, per number of exposures
        SN = self.SN([temp_rmag,zq,lobs_A])
        # scale with pixel width
        SN *= np.sqrt(pix_A)
        # now scale with number of exposures
        SN *= np.sqrt(1.0*Nexp/(1.0*self.file_Nexp))
        # in case SN is way too high, then one must prevent infinities
        SN = np.fmax(SN,1.0/large_noise)
        # finally return signal to noise
        return 1.0/(1.0*SN)

    def SmoothKernel_kms(self,z,pix_kms,res_kms,k_kms):
        # Convolution kernerl for the field (square this for power), including both pixelization and resolution
        x = np.fmax(0.5*k_kms*pix_kms,1.0e-10)
        kernelPixel = np.sin(x)/(1.0*x)
        kernelGauss = np.exp(-0.5*(k_kms*res_kms)**2)
        return kernelPixel* kernelGauss
        
    def SmoothKernel_kms_only_pix(self,z,pix_kms,k_kms):
        # Convolution kernel for the field, includes only pixelization, this is intended to be used for the 1D forecast
        x = np.fmax(0.5*k_kms*pix_kms,1.0e-10)
        kernelPixel = np.sin(x) / (1.0 * x)
        return kernelPixel
        
    
                
