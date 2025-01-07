import numpy as np
import QSO_lumin as lumin
import pixel as pix
import lya_convertor as convertor
import theory_P_lyas_arinyo as theory

"""
    Construct P_w^2D and P_N^eff in Equation (24) required for the total observed 3D lya power spectrum (Pw2D and PN_eff in function EffectiveDensityAndNoise())
    Calculate the variance of 3D lya power spectrum required for MCMC (function VarFluxP3D_Mpc_yao()) in Equation (22).
"""

class observed_3D(object):
    """
        lmin = 3501. + 200. * 10 -> z_mean = 3.61
        lmin = 3501. + 200. * 11 -> z_mean = 3.77
        lmin = 3501. + 200. * 12 -> z_mean = 3.94 
    """

        
    
    def __init__(self, params):
        # cosmology, important for models
        self.band = params['band']
        self.h = 0.6774
        self.c_kms = 2.998e5
        self.lya_A = 1215.67
        # some of the functionality of survey since it it more convenient
        # area of the sky that is covered (in ddeg^2)
        self.area_ddeg2 = 14000.0
        # wavelength range being covered (amrstrongs)
        self.lmin = 3501 #3501.0
        self.lmax = 3701 #3701.0
        # redshift range of the survey
        self.zq_min = 2.0
        self.zq_max = 4.0
        # now for the range in magnitudes
        self.mag_min = 16.5
        self.mag_max = 23.0
        # time to worry about the resolution aspects
        # pixel width in km/s
        self.pix_kms = 50.0
        # resolution in km/s
        self.res_kms = 70.0
        # where is the lya forest?
        self.l_forest_min = 985.0
        self.l_forest_max = 1200.0
        # signal to noise weights evaluated at this fourier mode??
        self.kt_w_deg = 7.0 # at approx. 0.1 h/Mpc, P: this doesn't really correspond to that k!
#        self.kt_w_deg = 2.078
        self.kp_w_kms = 0.001 # at approx. 0.1 h/Mpc
#        self.kp_w_kms = 0.01
        # for testing
        self.verbose = 1
        # get the other guys
        # luminosity function
        """ Now QLF needs to choose the default one (0) or the 'SV' leaked by Yeche (1) """
        QLF_verbose = 0 # 1 is SV
        self.QLF = lumin.QSO_LF(QLF_verbose)
        # spectrograph
        self.my_pix = pix.spec_pixel(self.band) # this is the original one
        # convertor
        self.convert = convertor.lya_convert()
        # theory
        self.my_P = theory.theory_P_lyas(params)
 
    def mean_z(self):
        # this function returns the central redshift covered in the bin
        # the wavelength range covered is
        l = np.sqrt(self.lmin*self.lmax) # the geometric mean because we want "same area" or inother words redshift are not being sum, but instead they correspond to products
        # find central redshift using the redshift definition
        z = l/(1.0 * 1215.67) - 1.0 # lya in A from wiki
        return z
           
    def L_kms(self):
        # returns the depth of the given redshift bin in km/s
        # the depth is given by e^L = dl/dL e^L = l - l_0 -> L = ln(l/l_0)
        L_kms = self.c_kms*np.log(self.lmax/(1.0 * self.lmin))
        return L_kms
           
    def Lq_kms(self):
        # computes the L_q needed for the P2w guy, it is the typical length of the Lya forest, here in km/s
        Lq_kms = self.c_kms * np.log(self.l_forest_max/(1.0 * self.l_forest_min))
        return Lq_kms
           
       # Besides these functions here, we will have the volume of the survey
    def Volume_Mpc(self,z):
        # gives the volume in Mpc^3, note that there is redshift dependence
        # survey volume in ddeg^2 * km/s
        V_degkms = self.area_ddeg2 * self.L_kms()
        # survey volume in Mpc^3
        V_Mpc = V_degkms * self.convert.dMpc_ddeg(z) * self.convert.dMpc_ddeg(z) / self.convert.dkms_dMpc(z)
        return V_Mpc

        
    def FluxP1D_kms(self,kp_kms):
        """1D Lya power spectrum in observed coordinates, smoothed with pixel width and resolution"""
        z = self.mean_z()
        # get P1D before smoothing
        P1D_kms = self.my_P.P1D_z_kms_PD2013(z,kp_kms)
        # smoothing (pixelization and resolution)
        Kernel = self.my_pix.SmoothKernel_kms(z,self.pix_kms,self.res_kms,kp_kms)
        P1D_kms = P1D_kms * Kernel * Kernel
        return P1D_kms
        
    def FluxP3D_Mpc(self,kt_deg,kp_kms):
        # note the funny observed units of the input, however output is going to be in Mpc^3
        # Also, here we just want to smooth things out with our spectrograph data
        z = self.mean_z()
        # transform Mpc to km/s
        dkms_dMpc = self.convert.dkms_dMpc(z)
        kp_Mpc = kp_kms * dkms_dMpc
        # transform from degrees
        dMpc_ddeg = self.convert.dMpc_ddeg(z)
        kt_Mpc = kt_deg / dMpc_ddeg
        # let's get mu
        k_Mpc = np.sqrt(kp_Mpc**2 + kt_Mpc**2)
        # take care of division by zero
        mu = kp_Mpc / (k_Mpc + 1.0e-10)
        # get power in Mpc. We will add reionization next
        P_Mpc = self.my_P.LyaLya_base_Mpc_norm(z,k_Mpc,mu)
        # add reio part
        """ Turn reio on or off """
        P_Mpc = P_Mpc + self.my_P.LyaLya_reio_Mpc_norm(z,k_Mpc,mu)
        P_degkms = P_Mpc * dkms_dMpc / (1.0 * dMpc_ddeg**2)
        Kernel = self.my_pix.SmoothKernel_kms(z, self.pix_kms, self.res_kms, kp_kms)
        P_degkms *= (Kernel**2)
        # finally recover my units
        P_Mpc = P_degkms * dMpc_ddeg**2 / (1.0 * dkms_dMpc)
        return P_Mpc
        

    def FluxP3D_degkms(self,kt_deg,kp_kms):
        # gets P3D in units of km/s deg, i.e. observed units
        z = self.mean_z()
        # get power
        P_Mpc = self.FluxP3D_Mpc(kt_deg,kp_kms)
        # get ready to transform
        dkms_dMpc = self.convert.dkms_dMpc(z)
        dMpc_ddeg = self.convert.dMpc_ddeg(z)
        P_degkms = P_Mpc * dkms_dMpc / (1.0 * dMpc_ddeg**2)
#        print('P_degkms = ',P_degkms)
        return P_degkms

    def I1_degkms(self,zq,mags,weights):
        # I1 integral, it represents an effective density of quasars and it depends on the current value of the weights that also in turn depend on I1. This is solve iteratively and converges fast
        # quasar number density
        dkms_dz = self.c_kms / (1.0 + zq)
        dndm_degdz = self.QLF.dNdzdmddeg2(zq,mags)
        # get rid of redshift
        dndm_degkms = dndm_degdz / (1.0 * dkms_dz)
        dm = mags[1] - mags[0]
        # weighted density of quasars
        I1 = np.sum(dndm_degkms * weights) * dm
        return I1

    def I2_degkms(self,zq,mags,weights):
        # I2 integral, sets the level of aliasing with P2w and P1D
        # quasar number density
        dndm_degdz = self.QLF.dNdzdmddeg2(zq,mags)
        dkms_dz = self.c_kms / (1.0 + zq)
        dndm_degkms = dndm_degdz / (1.0 * dkms_dz)
        dm = mags[1] - mags[0]
        I2 = np.sum(dndm_degkms * weights * weights) * dm
        return I2
 
    def VarN_m(self,zq,lc,mags):
        # noise pixel variance as function of magnitude. This is dimensionless!!
        z = lc / (1.0 * self.lya_A) - 1.0
        # pixel in Angstroms
        pix_A = self.pix_kms / self.convert.dkms_dlobs(z)
        # noise rms per pixel
        noise_rms = np.empty_like(mags)
        for i,m in enumerate(mags):
            noise_rms[i] = self.my_pix.PixelNoiseRMS(m,zq,lc,pix_A)
        noise_var = noise_rms**2
        return noise_var
 
    def I3_degkms(self,zq,lc,mags,weights):
        # I3 integral, it controls the effective noise power
        # pixel noise variance. This is dimensionless
        varN = self.VarN_m(zq,lc,mags)
        # quasar number density
        dndm_degdz = self.QLF.dNdzdmddeg2(zq,mags)
        dkms_dz = self.c_kms / (1.0 + zq)
        dndm_degkms = dndm_degdz / (1.0 * dkms_dz)
        dm = mags[1] - mags[0]
        I3 = np.sum(dndm_degkms * weights**2 * varN) * dm
        return I3
        
    def np_eff_degkms(self,zq,mags,weights):
        # effective density of pixels. It is used in constructing the weights as a function of mag
        # get effective density of quasars
        I1 = self.I1_degkms(zq,mags,weights)
        # number of pixels in a forest
        Npix = self.Lq_kms() / (1.0 * self.pix_kms)
        np_eff = I1 * Npix
        return np_eff
        
    def PN_m_degkms(self,zq,lc,mags,weights):
        # Effective noise power as a function of magnitude. This corresponds to as P_N(m). This is a proper 3D power, it is only used in constructing the weights
        # pixel noise variance (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # 3D effective density of pixels
        neff = self.np_eff_degkms(zq,mags,weights)
        PN = varN / (1.0 * neff)
        return PN

    def weights1(self,P3D_degkms,P1D_kms,zq,lc,mags,weights):
        # Function to compute new weights as a function of magnitude
        # 3D noise power as a function of magnitude
        PN = self.PN_m_degkms(zq,lc,mags,weights)
        # effective 3D density of quasars
        I1 = self.I1_degkms(zq,mags,weights)
        # 2D density of lines of sight in ddeg^-2
        n2D_los = I1 * self.Lq_kms()
        # weights include aliasing as signal
        PS = P3D_degkms + P1D_kms / (1.0 * n2D_los)
        weights = PS / (1.0 * (PS + PN))
        # let's check the weights
        if self.verbose > 2:
            print('P3D',P3D_degkms)
            print('P1D',P1D_kms)
            print('PN',PN)
            print('PS',PS)
            print('weights',weights)
        return weights
        
    def weights2(self,P3D_degkms,P1D_kms,zq,lc,mags,weights):
        # Code used in DESI, it computes nwe weights as a function of magnitude, using pixel variance
        # noise pixel variance as a function of magnitude (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # pixel variane from P1D also dimensionless
        var1D = P1D_kms / (1.0 * self.pix_kms)
        # effective density of pixels (3D)
        neff = self.np_eff_degkms(zq,mags,weights)
        # pixel variance from P3D (dimensionless)
        var3D = P3D_degkms * neff
        # signal variance (includes P1D and P3D)
        varS = var3D + var1D
        weights = varS / (1.0 * (varS + varN))
        if self.verbose > 2:
            print('P3D',P3D_degkms)
            print('P1D',P1D_kms)
            print('varN',varN)
            print('varS',varS)
            print('weights',weights)
        return weights
        
    def InitialWeights(self,P1D_kms,zq,lc,mags):
        # Computes initial weights as a function of magnitude
        # noise pixel variance as a function of magnitude (dimensionless)
        varN = self.VarN_m(zq,lc,mags)
        # pixel variance from P1D dimensionless
        var1D = P1D_kms / (1.0 * self.pix_kms)
        weights = var1D / (1.0 * (var1D + varN))
        if self.verbose > 2:
            print('P1D',P1D_kms)
            print('varN',varN)
            print('noise rms',np.sqrt(varN))
            print('var1D',var1D)
            print('weights',weights)
        return weights

    def ComputeWeights(self,P3D_degkms,P1D_kms,zq,lc,mags,Niter=3):
        # Computes the weights as a function of magnitude.
        # number of iterations 3
        # compute first weights using only 1D and noise variance
        weights = self.InitialWeights(P1D_kms,zq,lc,mags)
        weights2 = self.InitialWeights(P1D_kms,zq,lc,mags)
        for i in range(Niter):
            if self.verbose > 2:
                print(i,'<w>',np.mean(weights))
            weights = self.weights1(P3D_degkms,P1D_kms,zq,lc,mags,weights)
            # can also use the alternative form
            weights2 = self.weights2(P3D_degkms,P1D_kms,zq,lc,mags,weights2)
            if self.verbose > 2:
                print('weights',weights)
                print('weights2',weights2)
        return weights
        
    def EffectiveDensityAndNoise(self):
        # Compute effective density of lines of sight and effective noise power, i.e. terms Pw2D and PN_eff
        # mean wavelenght of bin
        lc = np.sqrt(self.lmin * self.lmax)
        # redshift of quasar for which forest is centered at z
        lrc = np.sqrt(self.l_forest_min * self.l_forest_max)
        zq = lc / (1.0 * lrc) - 1.0
        if self.verbose > 0:
            print('lc,lrc,zq=',lc,lrc,zq)
        # evaluate P1D and P3D for weighting
        P3D_w = self.FluxP3D_degkms(self.kt_w_deg,self.kp_w_kms)
        P1D_w = self.FluxP1D_kms(self.kp_w_kms)
        # The code below from Andreu evaluates all the quantities at the central redhift of the bin
        # and uses a single quasar redshift assuming that all pixels in the bin have restframe
        # wavelength of the center of the forest.
        # An alternative is to compute average over both redshift or absorption and over quasar redshift
        
        # set range of magnitudes used
        mmin = self.mag_min
        mmax = self.mag_max
        # binning
        dm = 0.025
        N = int((mmax - mmin) / (1.0 * dm))
        mags = np.linspace(mmin,mmax,N)
        # get weights iteratively
        weights = self.ComputeWeights(P3D_w,P1D_w,zq,lc,mags)
        # given weights compute integrals
        I1 = self.I1_degkms(zq,mags,weights)
        I2 = self.I2_degkms(zq,mags,weights)
        I3 = self.I3_degkms(zq,lc,mags,weights)
        if self.verbose > 0:
            print('I1, I2, I3 = ', I1,I2,I3)
            
        # length of forest in km/s
        Lq = self.Lq_kms()
        # length of pixel in km/s
        lp = self.pix_kms
        # effective 3D density of pixels
        np_eff = I1 * Lq / (1.0 * lp)
        # P2w
        Pw2D = I2 / (1.0 * I1**2 * Lq)
        # PNeff
        PN_eff = I3 * lp / (1.0 * I1**2 * Lq)
        if self.verbose > 0:
            print('np_eff, Pw2D, PN_eff = ', np_eff,Pw2D,PN_eff)
        return np_eff,Pw2D,PN_eff
        
    # Finally I can add all the stuff and construct the observed and compute the variance!!!!!!!!
    def TotalFluxP3D_degkms(self,kt_deg,kp_kms,Pw2D=None,PN_eff=None):
        # Sum of 3D Lya power with aliasing and noise
        # get redshift
        z = self.mean_z()
#        print z, self.lmin,self.lmax
        # the signal
        P3D = self.FluxP3D_degkms(kt_deg,kp_kms)
        # the aliasing term
        P1D = self.FluxP1D_kms(kp_kms)
        # check if they are given
        if not Pw2D or not PN_eff:
            # noise and rest of aliasing
            np_eff,Pw2D,PN_eff = self.EffectiveDensityAndNoise()
        # sum all the guys
        PT = P3D + Pw2D*P1D + PN_eff # the real observed P3D
#        print(P3D, Pw2D*P1D, PN_eff)
        return PT
        
    def TotalFluxP3D_Mpc(self,kt_deg,kp_kms,Pw2D=None,PN_eff=None):
        # Computes the observed 3D Lya power spectrum in Mpc^3
        z = self.mean_z()
        # get the total flux in funny units
        PT_degkms = self.TotalFluxP3D_degkms(kt_deg,kp_kms,Pw2D,PN_eff)
        # get ready to tranform to Mpc^3
        dkms_dMpc = self.convert.dkms_dMpc(z)
        dMpc_ddeg = self.convert.dMpc_ddeg(z)
#        print('PT_degkms, dkms_dMpc, dMpc_ddeg = ', PT_degkms, dkms_dMpc, dMpc_ddeg)
        PT = PT_degkms * dMpc_ddeg * dMpc_ddeg / (1.0 * dkms_dMpc)
        return PT
    
    # time for the variance
    def VarFluxP3D_Mpc(self, k_Mpc,mu,epsilon,dmu,Pw2D=None,PN_eff=None):
        """ now assumes log k bins """
        # Variance of 3D lya power spectrum in Mpc^3
        # range of mu is 0<mu<1
        # Note that we are adding the capability of input Pw2D and PN_eff to save some time since this does not depened on k
        # get redshift
        z = self.mean_z()
        # decompose into line of sight and transverse componentes (the k)
        kp_Mpc = k_Mpc * mu
        kt_Mpc = k_Mpc * np.sqrt(1.0 - mu**2)
        # transform from comoving to observed coordinates?
        dkms_dMpc = self.convert.dkms_dMpc(z)
        kp_kms = kp_Mpc / (1.0 * dkms_dMpc)
        dMpc_ddeg = self.convert.dMpc_ddeg(z)
        kt_deg = kt_Mpc * dMpc_ddeg
        # now total observed power in funny units
        totP_degkms = self.TotalFluxP3D_degkms(kt_deg,kp_kms,Pw2D,PN_eff)
        # back to Mpc^3
        totP_Mpc = totP_degkms * dMpc_ddeg* dMpc_ddeg / (1.0 * dkms_dMpc)
        # get volume of survey
        V_Mpc = self.Volume_Mpc(z)
        # Andreu: ' based on Eq. 8 in Seo & Eisenstein (2003), but with 0<mu<1 insted of -1<mu<1
#        Nmodes = V_Mpc * k_Mpc * k_Mpc * dk_Mpc * dmu / (2.0 * np.pi * np.pi) # original
        Nmodes = V_Mpc * k_Mpc * k_Mpc * k_Mpc * epsilon * dmu / (2.0 * np.pi * np.pi)
        varP = 2.0 * np.power(totP_Mpc,2) / (1.0 * Nmodes) # factor of 2 is because o<mu<1
        return varP


    def VarFluxP3D_Mpc_yao(self, k_Mpc,mu,dk,dmu,Pw2D=None,PN_eff=None):
        # Variance of 3D lya power spectrum in Mpc^3
        # range of mu is 0<mu<1
        # Note that we are adding the capability of input Pw2D and PN_eff to save some time since this does not depened on k
        # get redshift
        z = self.mean_z()
        # decompose into line of sight and transverse componentes (the k)
        kp_Mpc = k_Mpc * mu
        kt_Mpc = k_Mpc * np.sqrt(1.0 - mu**2)
        # transform from comoving to observed coordinates?
        dkms_dMpc = self.convert.dkms_dMpc(z)
        kp_kms = kp_Mpc / (1.0 * dkms_dMpc)
        dMpc_ddeg = self.convert.dMpc_ddeg(z)
        kt_deg = kt_Mpc * dMpc_ddeg
        # now total observed power in funny units
        totP_degkms = self.TotalFluxP3D_degkms(kt_deg,kp_kms,Pw2D,PN_eff)
        # back to Mpc^3
        totP_Mpc = totP_degkms * dMpc_ddeg* dMpc_ddeg / (1.0 * dkms_dMpc)
        # get volume of survey
        V_Mpc = self.Volume_Mpc(z)
        # Andreu: ' based on Eq. 8 in Seo & Eisenstein (2003), but with 0<mu<1 insted of -1<mu<1
#        Nmodes = V_Mpc * k_Mpc * k_Mpc * dk_Mpc * dmu / (2.0 * np.pi * np.pi) # original
        Nmodes = V_Mpc * k_Mpc * k_Mpc * dk * dmu / (2.0 * np.pi * np.pi)
        varP = 2.0 * np.power(totP_Mpc,2) / (1.0 * Nmodes) # factor of 2 is because o<mu<1
        return varP


    # need the capability of returning the total power spectrum for given 1D and 3D power spectrum
    # this will be done with new functions
    
    def Pw2D_to_neff_hMpc(self, Pw2D):
        """
            This function is useful to convert a Pw2D into the neff used in 1611.07527. Note that Pw2D has units of sq. deg and neff has h^-2 Mpc^2
        """
        z = self.mean_z()
        # first transform Pw2D to Mpc
        Pw2D_Mpc = Pw2D * self.convert.dMpc_ddeg(z)**2
        # deal with little h too
        Pw2D_hMpc = Pw2D_Mpc * self.h**2
        return 1. / Pw2D_hMpc
        
