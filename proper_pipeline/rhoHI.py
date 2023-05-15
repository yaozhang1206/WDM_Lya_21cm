import numpy as np
import pickle
import sys
from scipy import interpolate
from scipy import integrate
from scipy.interpolate import CloughTocher2DInterpolator
import f_mass
import Pm_DM
import scales
import ncdm


"""
    Build the HIdensity and we want to pickle this function to save time!
    
    Only needs to be run once per model!
    
"""
realization = sys.argv[1]
model = sys.argv[2]

directory = '/scratch/yaozhang/xray_%s/%s_%s/'%(realization,model,realization)

z_res = [6,7,8,9,10,11,12]

v_bc = 33

# dictionary
params={}

# set DM mass and sigma_8
# model should be like 3keV_s8, cdm_sminus...
if(model[0]=='3'):
    params['m_wdm'] = 3.0
elif(model[0]=='4'):
    params['m_wdm'] = 4.0
elif(model[0]=='6'):
    params['m_wdm'] = 6.0
elif(model[0]=='9'):
    params['m_wdm'] = 9.0
elif(model[0]=='c'):
    params['m_wdm'] = np.infty
else:
    print("DM mass cannot be determined.")
    sys.exit(1)

if(model[-3:]=='_s8'):
    params['sigma8'] = 0.8159
elif(model[-3:]=='nus'):
    params['sigma8'] = 0.7659
elif(model[-3:]=='lus'):
    params['sigma8'] = 0.8659
else:
    print("Sigma8 cannot be determined.")
    sys.exit(1)

params['h'] = 0.6774
params['Obh2'] = 0.02230
params['Och2'] = 0.1188
params['mnu'] = 0.194
#params['As'] = 2.142 # 10^9 * As    
params['ns'] = 0.9667
params['alphas'] = -0.002
params['taure'] = 0.066
params['Omega_cdm'] = 0.2602
#params['bHI'] = 2.82
#params['OHI'] = 1.18e-3 * 1.e3
#params['fast-model'] = 'cdm'
#params['fast-realization'] = 'ave'
#params['gadget-realization'] = 'ave'
#params['gadget-model'] = 'cdm'
#params['band'] = 'g'
params['sigma8'] = 0.8159   #Yao: note only one of As and sigma8 can be the input of CLASS dict
#params['telescope'] = 'skalow'
#params['t_int'] = 1000
#params['m_wdm'] = np.infty # for cdm is np.infty
#params['beam'] = 32 # think about this one
params['z_max_pk'] = 35 # 35 for running patchy class and 5.5 for everything else
params['P_k_max_1/Mpc'] = 110
#params['pickle'] = True # this needs to be True for this script.

# prepare for calculating halo mass function
Mhalos=np.logspace(7,16,10000) 
ks=np.logspace(-4,2,1000)   # in Mpc^-1
m_wdm = params['m_wdm']
# here we need CDM matter power spectrum because we first calculate CDM halo mass function
params['m_wdm'] = np.inf
Pm = Pm_DM.P_matter(params)  # Inputs: k [Mpc^-1], z; Outputs: P_m [Mpc^3]
params['m_wdm'] = m_wdm
print("m_wdm used in Pm: %.1f"%Pm.m_WDM_keV)
G=4.30091e-9
rho_crit=3*(params['h']*100)**2/8/np.pi/G*params['h'] # M_solar/(Mpc)^3/h
OMh2 = params['Obh2']+params['Och2']
f_b = params['Obh2'] / OMh2
rhom=rho_crit*OMh2/params['h']**2 #*(1+z)**3  #Yao: if not multiply (1+z)**3, it's comoving density

# prepare for converting from CDM to WDM halo mass function from Stucker et al. 2021
if m_wdm != np.inf:
    alpha = scales.alpha_wdm(mx=m_wdm, mode="schneider", omega_x=params['Omega_cdm'], h=params['h'], gx=1.5)
    alphanew, betanew, gammanew = scales.alpha_beta_gamma_3_to_2_par(alpha, 2.24, 5./1.12, gammamap=5.)
    mhm = scales.half_mode_mass(alphanew, betanew, gammanew, omega_m=0.3088) #Calculates the half mode mass in units of Msol/h


# fitting fomular from Tinker et al. 2008, eq.(2)
A=0.186
a=1.47
b=2.57
c=1.19

coords = []
rho_HI = []

# analyze directory by directory
for z_re in z_res:
    path = directory+"HMZ%02d/"%z_re
    a_list_file = path+'outputs_zre%d.txt'%z_re
    a_all = np.loadtxt(a_list_file)
    a_cs = a_all[:-1] # ignore z=2.5
    print("at z_re=%d:"%z_re)
    print(a_cs)
    z_cs = 1./a_cs-1
    c_s=[]
    gamma=[]
    T_0=[]
    logT_0=[]

    # when calculate c_s, we need all snapshots, in order to do interpolation and integration
    for i in range(len(a_cs)):
        f=f_mass.f_scale(path+'snapshot_%03d.extract'%i, a_cs[i], z_re)
        c_s_tmp,gamma_tmp,T_0_tmp,logT_0_tmp=f.cs_extract()
        c_s.append(c_s_tmp)
        gamma.append(gamma_tmp)
        T_0.append(T_0_tmp)
        logT_0.append(logT_0_tmp)
		
    np.savetxt(path+'cs_%02d.txt'%z_re, np.transpose([a_cs, z_cs, c_s, gamma, T_0, logT_0]))
    cs_func=interpolate.interp1d(np.array(a_cs),np.array(c_s), fill_value="extrapolate")


    MF=[]
    kF=[]
    kF_inverse2=[]
    kF_v_inverse2=[]
    kF_s_inverse2=[]

    for i in range(len(a_cs)):
        f=f_mass.f_scale(path+'snapshot_%03d.extract'%i, a_cs[i], z_re)
        MF_tmp,kF_v_inverse2_tmp,kF_s_inverse2_tmp,kF_tmp,kF_inverse2_tmp = f.results(cs_func,v_bc)
        MF.append(MF_tmp)
        kF_v_inverse2.append(kF_v_inverse2_tmp)
        kF_s_inverse2.append(kF_s_inverse2_tmp)
        kF.append(kF_tmp)
        kF_inverse2.append(kF_inverse2_tmp)
	
    np.savetxt(path+'f_scale_%02d.txt'%z_re, np.transpose([a_cs, z_cs, MF, kF, kF_inverse2, kF_v_inverse2, kF_s_inverse2]))
    


    # calculate halo mass function
    # Note we haven't implemented WDM here, we just use WDM power spectrum to calculate f(sigma)
    for i in range(len(z_cs)):
        sigmas=[]
        
        for m in Mhalos:
            R=(3.*m/4./np.pi/rhom)**(1./3)
            w_kR=3./(ks*R)**3*(np.sin(ks*R)-ks*R*np.cos(ks*R))
            d_sigma2 = []
            for j in range(len(ks)):
                d_sigma2.append(Pm.P_m_Mpc(ks[j], z_cs[i])*w_kR[j]**2*ks[j]**2/(2*np.pi**2))
            sigma_2=integrate.simps(d_sigma2,ks)
            sigma=np.sqrt(sigma_2)  
            sigmas.append(sigma)

        sigmas=np.array(sigmas)
        f_sigma=A*((sigmas/b)**(-a)+1)*np.exp(-c/sigmas**2)
        dndM= - f_sigma*rhom/Mhalos*np.gradient(np.log(sigmas),Mhalos)

        # convert from CDM to WDM halo mass function
        if m_wdm != np.inf:
        # convert from CDM to WDM halo mass function
            fhalos = []
            for m in Mhalos:
                fhalos.append(ncdm.mass_function_beta_mhm(m, beta=betanew, mhm=mhm, mode="halo"))
            fhalos = np.array(fhalos)
            dndM = dndM * fhalos
        M_baryon = np.array([f_b*M*(1+(2**(1./3)-1)*MF[i]/M)**(-3.) for M in Mhalos])
        # note MF[i] corresponds to z_cs[i]
        rho_HI.append(integrate.simps(dndM*M_baryon, Mhalos))
        coords.append([z_re, z_cs[i]])

print("coords:")
print(coords)
print("rho_HI")
print(rho_HI)

# finally we get all rho_HI at each z_re and z_obs!
# interpolate!

rho_func = interpolate.LinearNDInterpolator(coords, rho_HI)

# save it in pickle file
with open('../pickles/rho_HI_func/%s_%s.pkl'%(realization, model), 'wb') as f:
	pickle.dump(rho_func, f)
f.close()






        
        


