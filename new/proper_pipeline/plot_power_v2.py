import numpy as np
import theory_P_cross_full as theory_21
import theory_P_lyas_arinyo as theory_lya
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

def OHI_func(z):
        if 3.49 < z < 4.5:
            return 1.18e-3
        elif 4.5 <= z < 5.51:
            return 0.98e-3

def Tb_mean(z):

        return 27*np.sqrt((1+z)/10*0.15/0.1411)*(OHI_func(z)*0.6774**2/0.023)

# general dictionary
params={}
params['h'] = 0.6774
params['Obh2'] = 0.02230
params['Och2'] = 0.1188
params['mnu'] = 0.194
params['As'] = 2.142 # 10^9 * As
params['ns'] = 0.9667
params['alphas'] = -0.002
params['taure'] = 0.066
params['bHI'] = 2.82
params['OHI'] = 1.18e-3 * 1.e3
params['band'] = 'g'
params['telescope'] = 'skalow'
params['t_int'] = 1000
params['beam'] = 32 # think about this one
params['z_max_pk'] = 5.5 # only farmer would have 35 for running patchy class 
params['P_k_max_1/Mpc'] = 10
params['pickle'] = False # only farmer would have True here

params['sigma8'] = 0.8159

models = ['cdm_s8', '9keV_s8', '6keV_s8', '4keV_s8', '3keV_s8']
mwdm = [np.infty, 9.0, 6.0, 4.0, 3.0]
reals = ['r1', 'r2', 'r3', 'r4']

all_reals_lya = [[],[],[],[],[]]
all_reals_21 = [[],[],[],[],[]]

for i in range(5):
    params['m_wdm'] = mwdm[i]
    params['fast-model'] = models[i]
    params['gadget-model'] = models[i]    
    params['fast-realization'] = 'ave'
    params['gadget-realization'] = 'ave'
    all_reals_lya[i].append(theory_lya.theory_P_lyas(params))
    all_reals_21[i].append(theory_21.theory_P_cross(params))
    for j in range(4):
        params['fast-realization'] = reals[j]
        params['gadget-realization'] = 'ave'
        all_reals_lya[i].append(theory_lya.theory_P_lyas(params))
        all_reals_21[i].append(theory_21.theory_P_cross(params))
        params['fast-realization'] = 'ave'
        params['gadget-realization'] = reals[j]
        all_reals_lya[i].append(theory_lya.theory_P_lyas(params))
        all_reals_21[i].append(theory_21.theory_P_cross(params))


kbin = np.loadtxt('../data/21cmFAST/cross_21cm_ave_cdm_s8.txt', usecols=(1), unpack=True)

kbin = np.unique(kbin)
kbin = kbin[2:]
knum = len(kbin)
mu_21 = 0.9
mu_lya = 0.1
zbin_21 = [5.5, 3.5]
zbin_lya = [4., 2.]

data_21 = np.zeros((5,9,2,knum))
frac_21 = np.zeros((4,9,2,knum))
data_lya = np.zeros((5,9,2,knum))
frac_lya = np.zeros((4,9,2,knum))
for i in range(5):
    for j in range(9):
        for k in range(2):
            for q in range(knum):
                data_21[i,j,k,q] = all_reals_21[i][j].P3D_21_Mpc_norm(zbin_21[k], kbin[q], mu_21)
                data_lya[i,j,k,q] = all_reals_lya[i][j].LyaLya_base_Mpc_norm(zbin_lya[k], kbin[q], mu_lya) + all_reals_lya[i][j].LyaLya_reio_Mpc_norm(zbin_lya[k], kbin[q], mu_lya)
            data_21[i,j,k] = data_21[i,j,k] * Tb_mean(zbin_21[k]) ** 2

for i in range(4):
    frac_21[i,:,:,:] = (data_21[i+1,:,:,:] / data_21[0,:,:,:] -1.) * 100.
    frac_lya[i,:,:,:] = (data_lya[i+1,:,:,:] / data_lya[0,:,:,:] -1.) * 100.


frac_err_21 = np.zeros((4,2,knum))
frac_err_lya = np.zeros((4,2,knum))
error_21 = np.zeros((5,2,knum))
error_lya = np.zeros((5,2,knum))
for i in range(1,9):
    error_21 += (data_21[:,i,:,:] - data_21[:,0,:,:])**2
    error_lya += (data_lya[:,i,:,:] - data_lya[:,0,:,:])**2
    frac_err_21 += (frac_21[:,i,:,:] - frac_21[:,0,:,:])**2
    frac_err_lya += (frac_lya[:,i,:,:] - frac_lya[:,0,:,:])**2

error_21 = np.sqrt(error_21) / 2.
error_lya = np.sqrt(error_lya) / 2.
frac_err_21 = np.sqrt(frac_err_21) / 2.
frac_err_lya = np.sqrt(frac_err_lya) / 2.

cdm_norm_21 = np.zeros((2,knum))
cdm_norm_lya = np.zeros((2,knum))
for i in range(2):
    for j in range(knum):
        cdm_norm_21[i,j] = all_reals_21[0][0].HIHI_base_Mpc_norm(zbin_21[i], kbin[j], mu_21)
        cdm_norm_lya[i,j] = all_reals_lya[0][0].LyaLya_base_Mpc_norm(zbin_lya[i], kbin[j], mu_lya)

    cdm_norm_21[i] *= Tb_mean(zbin_21[i]) ** 2


labels = ['CDM', r'${\rm m_{WDM}=9\,keV}$',r'${\rm m_{WDM}=6\,keV}$', r'${\rm m_{WDM}=4\,keV}$', r'${\rm m_{WDM}=3\,keV}$']
colors = plt.cm.viridis(np.linspace(0,1,5))
fig, axs = plt.subplots(2,2,figsize=(10,10),sharex=True)
for i in range(2):
    for j in range(5):
        axs[i,0].fill_between(kbin, data_lya[j,0,i,:]-error_lya[j,i,:], data_lya[j,0,i,:]+error_lya[j,i,:],facecolor=colors[j], edgecolor=colors[j], label=labels[j])
        axs[i,1].fill_between(kbin, data_21[j,0,i,:]-error_21[j,i,:], data_21[j,0,i,:]+error_21[j,i,:],facecolor=colors[j], edgecolor=colors[j], label=labels[j])
        axs[i,0].plot(kbin, cdm_norm_lya[i], c=colors[0], ls='--')
        axs[i,1].plot(kbin, cdm_norm_21[i], c=colors[0], ls='--')



    axs[0,i].set_xscale('log')
    axs[0,i].set_xlim(0.05, 1.1)
    axs[1,i].set_xscale('log')
    axs[1,i].set_xlim(0.05, 1.1)
    axs[1,i].set_xlabel(r'$k\;{\rm (Mpc^{-1})}$', fontsize=16)
    axs[0,i].tick_params(axis = 'both', which = 'major', labelsize = 12)
    axs[1,i].tick_params(axis = 'both', which = 'major', labelsize = 12)
    axs[i,0].set_ylabel(r'$P_{\rm F}\;{\rm (Mpc^3)}$', fontsize=16)
    axs[i,1].set_ylabel(r'$P_{21}\;{\rm (mK^2Mpc^3)}$', fontsize=16)

left00, bottom00, width00, height00 = 0.26, 0.715, 0.21, 0.21
small_00 = fig.add_axes([left00, bottom00, width00, height00])
for j in [4,3,2,1]:
    small_00.fill_between(kbin, frac_lya[j-1,0,0,:]-frac_err_lya[j-1,0,:], frac_lya[j-1,0,0,:]+frac_err_lya[j-1,0,:],facecolor=colors[j], edgecolor=colors[j])
small_00.set_xlim(0.05, 1.1)
small_00.set_xscale('log')
small_00.set_xlabel(r'$k\;{\rm (Mpc^{-1})}$',labelpad=0)
small_00.set_ylabel(r'$(P_{\rm F}^{\rm WDM}-P_{\rm F}^{\rm CDM})/P_{\rm F}^{\rm CDM}\;(\%)$')
small_00.tick_params(axis = 'both', which = 'major')

left01, bottom01, width01, height01 = 0.76, 0.715, 0.21, 0.21
small_01 = fig.add_axes([left01, bottom01, width01, height01])
for j in [4,3,2,1]:
    small_01.fill_between(kbin, frac_21[j-1,0,0,:]-frac_err_21[j-1,0,:], frac_21[j-1,0,0,:]+frac_err_21[j-1,0,:],facecolor=colors[j], edgecolor=colors[j])
small_01.set_xlim(0.05, 1.1)
small_01.set_xscale('log')
small_01.set_xlabel(r'$k\;{\rm (Mpc^{-1})}$')
small_01.set_ylabel(r'$(P_{21}^{\rm WDM}-P_{21}^{\rm CDM})/P_{21}^{\rm CDM}\;(\%)$')
small_01.tick_params(axis = 'both', which = 'major')

left10, bottom10, width10, height10 = 0.26, 0.263, 0.21, 0.21
small_10 = fig.add_axes([left10, bottom10, width10, height10])
for j in [4,3,2,1]:
    small_10.fill_between(kbin, frac_lya[j-1,0,1,:]-frac_err_lya[j-1,1,:], frac_lya[j-1,0,1,:]+frac_err_lya[j-1,1,:],facecolor=colors[j], edgecolor=colors[j])
small_10.set_xlim(0.05, 1.1)
small_10.set_xscale('log')
small_10.set_xlabel(r'$k\;{\rm (Mpc^{-1})}$')
small_10.set_ylabel(r'$(P_{\rm F}^{\rm WDM}-P_{\rm F}^{\rm CDM})/P_{\rm F}^{\rm CDM}\;(\%)$')
small_10.tick_params(axis = 'both', which = 'major')

left11, bottom11, width11, height11 = 0.76, 0.263, 0.21, 0.21
small_11 = fig.add_axes([left11, bottom11, width11, height11])
for j in [4,3,2,1]:
    small_11.fill_between(kbin, frac_21[j-1,0,1,:]-frac_err_21[j-1,1,:], frac_21[j-1,0,1,:]+frac_err_21[j-1,1,:],facecolor=colors[j], edgecolor=colors[j])
small_11.set_xlim(0.05, 1.1)
small_11.set_xscale('log')
small_11.set_xlabel(r'$k\;{\rm (Mpc^{-1})}$')
small_11.set_ylabel(r'$(P_{21}^{\rm WDM}-P_{21}^{\rm CDM})/P_{21}^{\rm CDM}\;(\%)$')
small_11.tick_params(axis = 'both', which = 'major')

axs[0,0].set_title(r'${\rm Ly}\alpha$'+' forest power spectrum\nz=4.0', fontsize=14)
axs[0,1].set_title('21 cm IM power spectrum\nz=5.5', fontsize=14)
axs[1,0].set_title('z=2.0', fontsize=14)
axs[1,1].set_title('z=3.5', fontsize=14)
axs[1,1].set_ylim(0,6000)
axs[0,1].set_ylim(0,3500)
axs[0,0].set_ylim(0,600)
axs[1,0].set_ylim(0,35)
axs[0,0].legend(loc='lower right',fontsize=10, bbox_to_anchor=(0.985,0.05))
fig.tight_layout()
fig.savefig('power_spectrum_v2.pdf')



