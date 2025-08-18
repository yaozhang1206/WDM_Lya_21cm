import numpy as np
import theory_P_lyas_arinyo as theory
import observed_3D as obs
import matplotlib
from matplotlib import pyplot as plt
from scipy import interpolate


"""
    make the signal-to-noise plot for DESI and Stage V surveys
"""

next_gen = 0   # DESI
#next_gen = 1   # Stage V surveys

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.rm'] = 'Arial'


# general dictionary
params={}
params['h'] = 0.6774
params['Obh2'] = 0.02230
params['Och2'] = 0.1188
params['mnu'] = 0.194
params['ns'] = 0.9667
params['alphas'] = -0.002
params['taure'] = 0.066
params['bHI'] = 2.82
params['OHI'] = 1.18e-3 * 1.e3
params['fast-realization'] = 'ave'
params['gadget-realization'] = 'ave'
params['band'] = 'g'
params['telescope'] = 'skalow'
params['t_int'] = 1000
params['beam'] = 32 # think about this one
params['z_max_pk'] = 5.5 # only farmer would have 35 for running patchy class 
params['P_k_max_1/Mpc'] = 10
params['pickle'] = False # only farmer would have True here

# Yao: need to check the names of pickles and files

# prepare theoretical model for interpolation
params['sigma8'] = 0.8159
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.inf
cdm_s8 = theory.theory_P_lyas(params)

# fiducial model
params['sigma8'] = 0.8159
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.inf
ref = obs.observed_3D(params)

if next_gen > 0:
    ref.area_ddeg2 = 28000.0


# from wavelength range to z_bin
def obs_z(lmin, lmax):
        l_mean = np.sqrt(lmin * lmax) 
        z = l_mean / 1215.67 - 1.0
        return z

k_bin = np.linspace(0.06, 0.35, 30)
mu_bin = [0.875,0.625,0.375,0.125]

# case 1 z=2-4, 2 bins
# wavelength list
lmin_list = [3.*1215.67, 4.*1215.67]
lmax_list = [4.*1215.67, 5.*1215.67]
# bins to do summation
z_bin = [obs_z(lmin_list[i],lmax_list[i]) for i in range(2)]
znum = len(z_bin)
munum =  len(mu_bin)
knum = len(k_bin)

# calculate the bins by previous theoretical model
bins = np.zeros((znum,munum,knum))
for i in range(znum):
    for j in range(munum):
        for k in range(knum):
            bins[i,j,k] = cdm_s8.LyaLya_base_Mpc_norm(z_bin[i], k_bin[k], mu_bin[j]) + cdm_s8.LyaLya_reio_Mpc_norm(z_bin[i], k_bin[k], mu_bin[j])

# calculate bins of cdm reference model and variance
var_bin = np.zeros((znum,munum,knum))
for i in range(znum):
    z = z_bin[i]
    ref.lmin = lmin_list[i]
    ref.lmax = lmax_list[i]
    peff, pw, pn = ref.EffectiveDensityAndNoise()
    if next_gen > 0:
        pw /= 3.
        pn /= 3.
    for j in range(munum):
        for k in range(knum):
            var_bin[i,j,k] = ref.VarFluxP3D_Mpc_yao(k_bin[k], mu_bin[j], 0.01, 0.25, Pw2D=pw, PN_eff=pn)    # Yao: note that we use linear k bins here, not log k bins, so the calculation of Nmode need to be changed in observed_3D.py

noise = np.sqrt(var_bin)

lss = ['-','--']
colors = ['saddlebrown','peru','sandybrown','peachpuff']

lw = 1.5

i = 0
j = 0
fig = plt.figure(figsize=(6.5,6))
plt.plot(k_bin, bins[i,j]/noise[i,j],ls=lss[i],c=colors[j],label=r'$z=$'+'%.1f\n'%z_bin[i]+r'$\mu=$'+'%.3f'%mu_bin[j],lw=lw)
for j in range(1,munum):
        plt.plot(k_bin, bins[i,j]/noise[i,j],ls=lss[i],c=colors[j],label=r'$\mu=$'+'%.3f'%mu_bin[j],lw=lw)
i = 1
j = 0
plt.plot(k_bin, bins[i,j]/noise[i,j],ls=lss[i],c=colors[j],label=r'$z=$'+'%.1f'%z_bin[i],lw=lw)
for j in range(1,munum):
        plt.plot(k_bin, bins[i,j]/noise[i,j],ls=lss[i],c=colors[j],lw=lw)

plt.xlim(0.06,0.35)
plt.xscale('log')
plt.xlabel(r'$k\ {\rm (Mpc^{-1})}$', fontsize=14)
plt.ylabel(r'$P_{\rm F}^{\rm fid}/ \sigma_{\rm F}$', fontsize=14)
plt.xticks([0.06, 0.1, 0.2, 0.3],labels=['0.06', '0.1', '0.2', '0.3'])
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=12)
fig.tight_layout()
fig.savefig('SNR.pdf',bbox_inches="tight")

