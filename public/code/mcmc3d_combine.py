import numpy as np
import theory_P_21cm as theory_21
import theory_P_lyas_arinyo as theory_lya
import observed_3D as obs
import wedge
import emcee
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from scipy import interpolate
import corner
import time
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

start = time.time()

tele = sys.argv[1]

# general dictionary
params={}
params['h'] = 0.6774
params['Obh2'] = 0.02230
params['Och2'] = 0.1188
params['ns'] = 0.9667
params['mnu'] = 0.194
params['As'] = 2.142 # 10^9 * As
params['alphas'] = -0.002
params['taure'] = 0.066
params['bHI'] = 2.82
params['OHI'] = 1.18e-3 * 1.e3
params['fast-realization'] = 'r1'
params['gadget-realization'] = 'r1'
params['band'] = 'g'
params['telescope'] = tele
params['beam'] = 32 # think about this one
params['z_max_pk'] = 5.5 # only farmer would have 35 for running patchy class 
params['P_k_max_1/Mpc'] = 10
params['pickle'] = False # only farmer would have True here

# Yao: need to check the names of pickles and files

if params['telescope'] == 'skalow':
    D_dish = 40.0 # meter
    params['t_int'] = 5000. # hours
    z_bin_21 = [3.65, 3.95, 4.25, 4.55, 4.85, 5.15, 5.45]
    dz_21 = 0.3
elif params['telescope'] == 'puma':
    D_dish = 6.0
    params['t_int'] = 1000.
    z_bin_21 = [3.6+0.2*i for i in range(10)]
    dz_21 = 0.2
else:
    print("Cannot determine the telescope!")
    exit(1)

# prepare theoretical model for interpolation
params['sigma8'] = 0.8159

params['fast-model'] = '3keV_s8'
params['gadget-model'] = '3keV_s8'
params['m_wdm'] = 3.0
wdm_3keV_s8_lya = theory_lya.theory_P_lyas(params)
wdm_3keV_s8_21 = theory_21.theory_P_21(params)

params['fast-model'] = '4keV_s8'
params['gadget-model'] = '4keV_s8'
params['m_wdm'] = 4.0
wdm_4keV_s8_lya = theory_lya.theory_P_lyas(params)
wdm_4keV_s8_21 = theory_21.theory_P_21(params)

params['fast-model'] = '6keV_s8'
params['gadget-model'] = '6keV_s8'
params['m_wdm'] = 6.0
wdm_6keV_s8_lya = theory_lya.theory_P_lyas(params)
wdm_6keV_s8_21 = theory_21.theory_P_21(params)

params['fast-model'] = '9keV_s8'
params['gadget-model'] = '9keV_s8'
params['m_wdm'] = 9.0
wdm_9keV_s8_lya = theory_lya.theory_P_lyas(params)
wdm_9keV_s8_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.infty
cdm_s8_lya = theory_lya.theory_P_lyas(params)
cdm_s8_21 = theory_21.theory_P_21(params)

params['sigma8'] = 0.8659

params['fast-model'] = '3keV_splus'
params['gadget-model'] = '3keV_splus'
params['m_wdm'] = 3.0
wdm_3keV_splus_lya = theory_lya.theory_P_lyas(params)
wdm_3keV_splus_21 = theory_21.theory_P_21(params)

params['fast-model'] = '4keV_splus'
params['gadget-model'] = '4keV_splus'
params['m_wdm'] = 4.0
wdm_4keV_splus_lya = theory_lya.theory_P_lyas(params)
wdm_4keV_splus_21 = theory_21.theory_P_21(params)

params['fast-model'] = '6keV_splus'
params['gadget-model'] = '6keV_splus'
params['m_wdm'] = 6.0
wdm_6keV_splus_lya = theory_lya.theory_P_lyas(params)
wdm_6keV_splus_21 = theory_21.theory_P_21(params)

params['fast-model'] = '9keV_splus'
params['gadget-model'] = '9keV_splus'
params['m_wdm'] = 9.0
wdm_9keV_splus_lya = theory_lya.theory_P_lyas(params)
wdm_9keV_splus_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'cdm_splus'
params['gadget-model'] = 'cdm_splus'
params['m_wdm'] = np.infty
cdm_splus_lya = theory_lya.theory_P_lyas(params)
cdm_splus_21 = theory_21.theory_P_21(params)

params['sigma8'] = 0.7659

params['fast-model'] = '3keV_sminus'
params['gadget-model'] = '3keV_sminus'
params['m_wdm'] = 3.0
wdm_3keV_sminus_lya = theory_lya.theory_P_lyas(params)
wdm_3keV_sminus_21 = theory_21.theory_P_21(params)

params['fast-model'] = '4keV_sminus'
params['gadget-model'] = '4keV_sminus'
params['m_wdm'] = 4.0
wdm_4keV_sminus_lya = theory_lya.theory_P_lyas(params)
wdm_4keV_sminus_21 = theory_21.theory_P_21(params)

params['fast-model'] = '6keV_sminus'
params['gadget-model'] = '6keV_sminus'
params['m_wdm'] = 6.0
wdm_6keV_sminus_lya = theory_lya.theory_P_lyas(params)
wdm_6keV_sminus_21 = theory_21.theory_P_21(params)

params['fast-model'] = '9keV_sminus'
params['gadget-model'] = '9keV_sminus'
params['m_wdm'] = 9.0
wdm_9keV_sminus_lya = theory_lya.theory_P_lyas(params)
wdm_9keV_sminus_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'cdm_sminus'
params['gadget-model'] = 'cdm_sminus'
params['m_wdm'] = np.infty
cdm_sminus_lya = theory_lya.theory_P_lyas(params)
cdm_sminus_21 = theory_21.theory_P_21(params)

# add zeta
params['m_wdm'] = np.infty

params['sigma8'] = 0.7659
params['fast-model'] = 'zeta_p1'
params['gadget-model'] = 'zeta_p1'
zeta_p1_lya = theory_lya.theory_P_lyas(params)
zeta_p1_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'zeta_p2'
params['gadget-model'] = 'zeta_p2'
zeta_p2_lya = theory_lya.theory_P_lyas(params)
zeta_p2_21 = theory_21.theory_P_21(params)

params['sigma8'] = 0.8659
params['fast-model'] = 'zeta_p3'
params['gadget-model'] = 'zeta_p3'
zeta_p3_lya = theory_lya.theory_P_lyas(params)
zeta_p3_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'zeta_p4'
params['gadget-model'] = 'zeta_p4'
zeta_p4_lya = theory_lya.theory_P_lyas(params)
zeta_p4_21 = theory_21.theory_P_21(params)

params['m_wdm'] = 3.0

params['sigma8'] = 0.7659
params['fast-model'] = 'zeta_p5'
params['gadget-model'] = 'zeta_p5'
zeta_p5_lya = theory_lya.theory_P_lyas(params)
zeta_p5_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'zeta_p6'
params['gadget-model'] = 'zeta_p6'
zeta_p6_lya = theory_lya.theory_P_lyas(params)
zeta_p6_21 = theory_21.theory_P_21(params)

params['sigma8'] = 0.8659
params['fast-model'] = 'zeta_p7'
params['gadget-model'] = 'zeta_p7'
zeta_p7_lya = theory_lya.theory_P_lyas(params)
zeta_p7_21 = theory_21.theory_P_21(params)

params['fast-model'] = 'zeta_p8'
params['gadget-model'] = 'zeta_p8'
zeta_p8_lya = theory_lya.theory_P_lyas(params)
zeta_p8_21 = theory_21.theory_P_21(params)


# !! we use 1/m and sigma8 as parameter !!
inverse_mass = [1/3., 1/4., 1/6., 1/9., 0.]
sigma8 = [0.7659, 0.8159, 0.8659]

coords = np.zeros((23,3))

for i in range(3):
    for j in range(5):
        coords[5*i+j]= [inverse_mass[j], sigma8[i], 30.]

coords[15] = [0., 0.7659, 20.0]
coords[16] = [0., 0.7659, 40.0]
coords[17] = [0., 0.8659, 20.0]
coords[18] = [0., 0.8659, 40.0]
coords[19] = [1./3., 0.7659, 20.0]
coords[20] = [1./3., 0.7659, 40.0]
coords[21] = [1./3., 0.8659, 20.0]
coords[22] = [1./3., 0.8659, 40.0]

#*********** Lya forest *********#

# fiducial model for lya
params['sigma8'] = 0.8159
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.infty
ref_lya = obs.observed_3D(params)

'''
Yao: Note that PUMA is ~2040 instrument, so at that time Lya survey will be better than DESI
So we assume a DESI++ instrument, i.e. survey area = 2 * DESI's = 28,000 deg^2
Reduce noise of the instrument and get more quasars, P_N and P_w can be reduced by a factor of 3.
'''
if tele == 'puma':
    ref_lya.area_ddeg2 = 28000.


# from wavelength range to z_bin
def obs_z(l):
        l_mean = np.sqrt(l * (l + 200.0)) 
        z = l_mean / 1215.67 - 1.0
        return z

# wavelength list
lmin_list = [3501.0 + i * 200.0 for i in range(13)]

# bins for lya
z_bin_lya = [obs_z(l) for l in lmin_list]
k_bin_lya = np.linspace(0.06, 0.45, 40)
mu_bin_lya = [0.1, 0.3, 0.5, 0.7, 0.9]

bins_lya = np.zeros((len(z_bin_lya)*len(k_bin_lya)*len(mu_bin_lya), 23))
i = 0
for z in z_bin_lya:
    for k in k_bin_lya:
        for mu in mu_bin_lya:
            bins_lya[i,0] = wdm_3keV_sminus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_3keV_sminus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,1] = wdm_4keV_sminus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_4keV_sminus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,2] = wdm_6keV_sminus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_6keV_sminus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,3] = wdm_9keV_sminus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_9keV_sminus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,4] = cdm_sminus_lya.LyaLya_base_Mpc_norm(z, k, mu) + cdm_sminus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,5] = wdm_3keV_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_3keV_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,6] = wdm_4keV_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_4keV_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,7] = wdm_6keV_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_6keV_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,8] = wdm_9keV_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_9keV_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,9] = cdm_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + cdm_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,10] = wdm_3keV_splus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_3keV_splus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,11] = wdm_4keV_splus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_4keV_splus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,12] = wdm_6keV_splus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_6keV_splus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,13] = wdm_9keV_splus_lya.LyaLya_base_Mpc_norm(z, k, mu) + wdm_9keV_splus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,14] = cdm_splus_lya.LyaLya_base_Mpc_norm(z, k, mu) + cdm_splus_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,15] = zeta_p1_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p1_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,16] = zeta_p2_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p2_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,17] = zeta_p3_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p3_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,18] = zeta_p4_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p4_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,19] = zeta_p5_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p5_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,20] = zeta_p6_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p6_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,21] = zeta_p7_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p7_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            bins_lya[i,22] = zeta_p8_lya.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p8_lya.LyaLya_reio_Mpc_norm(z, k, mu)
            i += 1

# time to interpolate!
bins_inter_lya = []

for i in range(len(bins_lya)):
    bins_inter_lya.append(interpolate.LinearNDInterpolator(coords, bins_lya[i]))

ref_bin_lya = []
var_bin_lya = []
for i in range(len(z_bin_lya)):
    z = z_bin_lya[i]
    ref_lya.lmin = lmin_list[i]
    ref_lya.lmax = lmin_list[i] + 200.0
    peff, pw, pn = ref_lya.EffectiveDensityAndNoise() # for each redshift, we calculate Pw2D and PN_eff once to save some time
    # for puma, we assume it combines with DESI++ instruments, and P_N and P_w can be reduced by a factor of 3
    if tele == 'puma':
        pw /= 3.
        pn /= 3.
    for k in k_bin_lya:
        for mu in mu_bin_lya:
            #kp_kms = k * mu / dkms_dmpc
            #kt_deg = k * np.sqrt(1.0 - mu**2) * dmpc_ddeg
            ref_bin_lya.append(cdm_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + cdm_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu))
            var_bin_lya.append(ref_lya.VarFluxP3D_Mpc_yao(k, mu, 0.01, 0.2, Pw2D=pw, PN_eff=pn))    # Yao: note that we use linear k bins here, not log k bins, so the calculation of Nmode need to be changed in observed_3D.py


#******** 21cm ********#
bins_inter_21 = []
ref_bin_21 = []
var_bin_21 = []

bin_class = wedge.bins(dish_D=D_dish)
for z in z_bin_21:
    bin_class.z = z
    bin_class.dz = dz_21
    k_bin, dk_bin, kmin = bin_class.k_bins()
    k_parallel_min = bin_class.k_parallel_min()
    k_perp_min = bin_class.k_perp_min()
    if tele == 'skalow':
        mu_bin = [0.1, 0.3, 0.5, 0.7, 0.9]
        dmu = 0.2
    elif tele == 'puma':
        mu_bin, dmu = bin_class.mu_bins()

    for k, dk in zip(k_bin, dk_bin):
        for mu in mu_bin:
            k_parallel = k * mu
            k_perp = k * np.sqrt(1-mu**2)
            if (k_parallel<k_parallel_min) or (k_perp<k_perp_min):
                continue
            one_bin = np.zeros(23)
            one_bin[0] = wdm_3keV_sminus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[1] = wdm_4keV_sminus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[2] = wdm_6keV_sminus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[3] = wdm_9keV_sminus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[4] = cdm_sminus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[5] = wdm_3keV_s8_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[6] = wdm_4keV_s8_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[7] = wdm_6keV_s8_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[8] = wdm_9keV_s8_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[9] = cdm_s8_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[10] = wdm_3keV_splus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[11] = wdm_4keV_splus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[12] = wdm_6keV_splus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[13] = wdm_9keV_splus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[14] = cdm_splus_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[15] = zeta_p1_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[16] = zeta_p2_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[17] = zeta_p3_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[18] = zeta_p4_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[19] = zeta_p5_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[20] = zeta_p6_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[21] = zeta_p7_21.P3D_21_Mpc_norm(z, k, mu)
            one_bin[22] = zeta_p8_21.P3D_21_Mpc_norm(z, k, mu)
            bins_inter_21.append(interpolate.LinearNDInterpolator(coords, one_bin))
            ref_bin_21.append(cdm_s8_21.P3D_21_Mpc_norm(z, k, mu))
            var_bin_21.append(cdm_s8_21.Var_autoHI_Mpc_yao(z, k, mu, dz_21, dk, dmu)) # Yao: note this func assume dmu=0.2, which might need modification



end1 = time.time()
interp_time = end1 - start
print("Preparing models took {0:.1f} seconds".format(interp_time))


# the time taken for each likelihood calculation
cal_time = []
#  log-probability function
def log_prob(theta):
    cal_start = time.time()
    inver_mass, sigma, zeta = theta
    test = (bins_inter_lya[1])(inver_mass,sigma,zeta)
    if (np.isnan(test)):
        return -np.inf, -np.inf
    else:
        log_p = 0
        for i in range(len(bins_inter_lya)):
            log_p += ((bins_inter_lya[i])(inver_mass,sigma,zeta) - ref_bin_lya[i])**2 / var_bin_lya[i]

        for i in range(len(bins_inter_21)):
            log_p += ((bins_inter_21[i])(inver_mass,sigma,zeta) - ref_bin_21[i])**2 / var_bin_21[i]
        cal_end = time.time()
        cal_time.append(cal_end - cal_start)
        if (np.isnan(log_p)):
            return -np.inf, -np.inf
        return -0.5 * log_p, 0.0


nw = 32
nd = 3

# we need to make the initial value in the prior range
initial = np.zeros((nw, nd))
for l in range(nw):
    initial[l,0] = np.random.rand()/3.
    initial[l,1] = 0.7659 + np.random.rand() * 0.1
    initial[l,2] = 20.0 + np.random.rand() * 20.0
print(initial)

# run mcmc chain
filename = "chain_combine_%s_zeta_pop2.h5"%(tele)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, backend=backend, moves=emcee.moves.StretchMove(a=4.0))
sampler.run_mcmc(initial, 50000, progress=False)

end2 = time.time()
mcmc_time = end2 - end1
print("MCMC took {0:.1f} seconds".format(mcmc_time))

fig, axs = plt.subplots(3)
samples1 = sampler.get_chain()
axs[0].plot(range(len(samples1)), samples1[:, :, 0], "k", alpha=0.3)
axs[0].set_xlabel('step number')
axs[0].set_ylabel('1keV / m')
axs[1].plot(range(len(samples1)), samples1[:, :, 1], "k", alpha=0.3)
axs[1].set_xlabel('step number')
axs[1].set_ylabel(r'$\sigma_8$')
axs[2].plot(range(len(samples1)), samples1[:, :, 2], "k", alpha=0.3)
axs[2].set_xlabel('step number')
axs[2].set_ylabel(r'$\zeta$')
fig.savefig('chain_combine_%s_zeta_pop2.pdf'%(tele))


samples = sampler.get_chain(flat=True, discard=2000, thin=20)
fig1 = corner.corner(
    samples, labels=['1keV / m', r'$\sigma_8$', r'$\zeta$'], truths=[0,0.8159,30.])
fig1.savefig('corner_combine_%s_zeta_pop2.pdf'%(tele))

cal_time = np.array(cal_time, dtype=float)
print('calculation time for each step: ', np.mean(cal_time))
print("length: ", len(cal_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))

