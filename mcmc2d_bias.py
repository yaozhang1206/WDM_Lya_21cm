import numpy as np
import theory_P_lyas_fur_yao_bias_version as theory
import observed_3D_yao_bias_version as obs
import emcee
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from scipy import interpolate
import h5py
import corner
from multiprocessing import Pool
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"

start = time.time()

# prepare theoretical model for interpolation
wdm_3keV_s8 = theory.theory_P_lyas(3.0, 0.8159, '21cm_avg_wdm3', 'gadget_ave_3keV')
wdm_4keV_s8 = theory.theory_P_lyas(4.0, 0.8159, '21cm_avg_wdm4', 'gadget_ave_4keV')
wdm_6keV_s8 = theory.theory_P_lyas(6.0, 0.8159, '21cm_avg_wdm6', 'gadget_ave_6keV')
wdm_9keV_s8 = theory.theory_P_lyas(9.0, 0.8159, '21cm_avg_wdm9', 'gadget_ave_9keV')
cdm_s8 = theory.theory_P_lyas(np.infty, 0.8159, '21cm_avg_cdm', 'gadget_ave_cdm')
wdm_3keV_splus = theory.theory_P_lyas(3.0, 0.8659, '21cm_wdm3_s8plus_r1', 'gadget_sigma+0.05_ave_3keV')
wdm_4keV_splus = theory.theory_P_lyas(4.0, 0.8659, '21cm_wdm4_s8plus_r1', 'gadget_sigma+0.05_ave_4keV')
wdm_6keV_splus = theory.theory_P_lyas(6.0, 0.8659, '21cm_wdm6_s8plus_r1', 'gadget_sigma+0.05_ave_6keV')
wdm_9keV_splus = theory.theory_P_lyas(9.0, 0.8659, '21cm_wdm9_s8plus_r1', 'gadget_sigma+0.05_ave_9keV')
cdm_splus = theory.theory_P_lyas(np.infty, 0.8659, '21cm_cdm_s8plus_r1', 'gadget_sigma+0.05_ave_cdm')
wdm_3keV_sminus = theory.theory_P_lyas(3.0, 0.7659, '21cm_wdm3_s8minus_r1', 'gadget_sigma-0.05_ave_3keV')
wdm_4keV_sminus = theory.theory_P_lyas(4.0, 0.7659, '21cm_wdm4_s8minus_r1', 'gadget_sigma-0.05_ave_4keV')
wdm_6keV_sminus = theory.theory_P_lyas(6.0, 0.7659, '21cm_wdm6_s8minus_r1', 'gadget_sigma-0.05_ave_6keV')
wdm_9keV_sminus = theory.theory_P_lyas(9.0, 0.7659, '21cm_wdm9_s8minus_r1', 'gadget_sigma-0.05_ave_9keV')
cdm_sminus = theory.theory_P_lyas(np.infty, 0.7659, '21cm_cdm_s8minus_r1', 'gadget_sigma-0.05_ave_cdm')

# observed CDM REFERENCE model, "average" in gaussian likelihood
ref = obs.observed_3D('g')

# from wavelength range to z_bin
def obs_z(l):
        l_mean = np.sqrt(l * (l + 200.0)) 
        z = l_mean / 1215.67 - 1.0
        return z

# wavelength list
lmin_list = [3501.0 + i * 200.0 for i in range(12)]
# bins to do summation
z_bin = [obs_z(l) for l in lmin_list]
k_bin = np.linspace(0.06, 0.85, 80)
mu_bin = [0.1, 0.3, 0.5, 0.7, 0.9]
'''
z_bin = [2.0, 3.0]
k_bin = [0.08, 0.1]
mu_bin = [0.1, 0.3]
'''
# calculate the bins by previous theoretical model
bins = np.zeros((len(z_bin)*len(k_bin)*len(mu_bin), 3, 5))
i = 0
for z in z_bin:
    for k in k_bin:
        for mu in mu_bin:
            bins[i,0,0] = wdm_3keV_sminus.FluxP3D_lya_Mpc(z, k, mu) + wdm_3keV_sminus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,0,1] = wdm_4keV_sminus.FluxP3D_lya_Mpc(z, k, mu) + wdm_4keV_sminus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,0,2] = wdm_6keV_sminus.FluxP3D_lya_Mpc(z, k, mu) + wdm_6keV_sminus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,0,3] = wdm_9keV_sminus.FluxP3D_lya_Mpc(z, k, mu) + wdm_9keV_sminus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,0,4] = cdm_sminus.FluxP3D_lya_Mpc(z, k, mu) + cdm_sminus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,1,0] = wdm_3keV_s8.FluxP3D_lya_Mpc(z, k, mu) + wdm_3keV_s8.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,1,1] = wdm_4keV_s8.FluxP3D_lya_Mpc(z, k, mu) + wdm_4keV_s8.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,1,2] = wdm_6keV_s8.FluxP3D_lya_Mpc(z, k, mu) + wdm_6keV_s8.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,1,3] = wdm_9keV_s8.FluxP3D_lya_Mpc(z, k, mu) + wdm_9keV_s8.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,1,4] = cdm_s8.FluxP3D_lya_Mpc(z, k, mu) + cdm_s8.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,2,0] = wdm_3keV_splus.FluxP3D_lya_Mpc(z, k, mu) + wdm_3keV_splus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,2,1] = wdm_4keV_splus.FluxP3D_lya_Mpc(z, k, mu) + wdm_4keV_splus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,2,2] = wdm_6keV_splus.FluxP3D_lya_Mpc(z, k, mu) + wdm_6keV_splus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,2,3] = wdm_9keV_splus.FluxP3D_lya_Mpc(z, k, mu) + wdm_9keV_splus.FluxP3D_reio_Mpc(z, k, mu)
            bins[i,2,4] = cdm_splus.FluxP3D_lya_Mpc(z, k, mu) + cdm_splus.FluxP3D_reio_Mpc(z, k, mu)
            i += 1

bins_save = np.reshape(bins, (len(z_bin)*len(k_bin)*len(mu_bin),15))
np.savetxt('theory_bins_bias.txt', bins_save, fmt='%.5f', delimiter=' ')

# !! we use 1/m and sigma8 as parameter !!
inverse_mass = [1/3, 1/4, 1/6, 1/9, 0]
sigma8 = [0.7659, 0.8159, 0.8659]

# time to interpolate!
bins_inter = []

for i in range(len(bins)):
    bins_inter.append(interpolate.interp2d(inverse_mass, sigma8, bins[i]))

end1 = time.time()
interp_time = end1 - start
print("Interpolation took {0:.1f} seconds".format(interp_time))

# calculate bins of observed cdm reference model and variance
ref_bin = []
var_bin = []
for i in range(len(z_bin)):
    z = z_bin[i]
    ref.lmin = lmin_list[i]
    ref.lmax = lmin_list[i] + 200.0
    peff, pw, pn = ref.EffectiveDensityAndNoise() # for each redshift, we calculate Pw2D and PN_eff once to save some time
    #dkms_dmpc = ref.convert.dkms_dMpc(z)          # convert to funny units
    #dmpc_ddeg = ref.convert.dMpc_ddeg(z)
    for k in k_bin:
        for mu in mu_bin:
            #kp_kms = k * mu / dkms_dmpc
            #kt_deg = k * np.sqrt(1.0 - mu**2) * dmpc_ddeg
            ref_bin.append(cdm_s8.FluxP3D_lya_Mpc(z, k, mu) + cdm_s8.FluxP3D_reio_Mpc(z, k, mu))
            var_bin.append(ref.VarFluxP3D_Mpc(k, mu, 0.01, 0.2, Pw2D=pw, PN_eff=pn))

ref_save = np.zeros((len(ref_bin),2))
for i in range(len(ref_bin)):
    ref_save[i,0] = ref_bin[i]
    ref_save[i,1] = var_bin[i]
np.savetxt('ref_bins_bias.txt', ref_save, fmt='%.5f', delimiter=' ')

end2 = time.time()
ref_time = end2 - end1
print("Reference preparation took {0:.1f} seconds".format(ref_time))


# the time taken for each likelihood calculation
cal_time = []

#  log-probability function
def log_prob(theta, ref, var):
    cal_start = time.time()
    inver_mass, sigma = theta
    if inver_mass > 1/3 or inver_mass < 0 or sigma < 0.7659 or sigma > 0.8659:
        return -np.inf, -np.inf
    else:
        log_p = 0
        for i in range(len(bins_inter)):
            log_p += ((bins_inter[i])(inver_mass,sigma) - ref[i])**2 / var[i]
        cal_end = time.time()
        cal_time.append(cal_end - cal_start)
        return -0.5 * log_p, 0.0

nw = 32
nd = 2

# we need to make the initial value in the prior range
initial = np.zeros((nw, nd))
for l in range(nw):
    initial[l,0] = np.random.rand()/3.0
    initial[l,1] = 0.7659 + np.random.rand() * 0.1
print(initial)

# run mcmc chain
filename = "chain2d_bias.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, args=(ref_bin, var_bin), backend=backend)
sampler.run_mcmc(initial, 3000, progress=True)

end3 = time.time()
mcmc_time = end3 - end2
print("MCMC took {0:.1f} seconds".format(mcmc_time))

fig, axs = plt.subplots(2)
samples1 = sampler.get_chain()
axs[0].plot(range(len(samples1)), samples1[:, :, 0], "k", alpha=0.3)
axs[0].set_xlabel('step number')
axs[0].set_ylabel('1keV / m')
axs[1].plot(range(len(samples1)), samples1[:, :, 1], "k", alpha=0.3)
axs[1].set_xlabel('step number')
axs[1].set_ylabel('sigma 8')
fig.savefig('chain2d_bias.pdf')

# length: > 50 * autocorrelation time;  discard: a few times autocorrelation time;  thin:  half the autocorrelation time

samples = sampler.get_chain(flat=True, discard=200, thin=20)
fig1 = corner.corner(
    samples, labels=['1keV / m', 'sigma8'], truths=[0, 0.8159])
fig1.savefig('corner_bias.pdf')

cal_time = np.array(cal_time, dtype=float)
print('calculation time for each step: ', np.mean(cal_time))
print("length: ", len(cal_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
