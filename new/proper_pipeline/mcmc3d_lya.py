import numpy as np
import theory_P_lyas_arinyo as theory
import observed_3D as obs
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

next_gen = int(sys.argv[1])

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
params['fast-realization'] = 'r1'   # yao: need to be fixed later
params['gadget-realization'] = 'r1'
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

params['fast-model'] = '3keV_s8'
params['gadget-model'] = '3keV_s8'
params['m_wdm'] = 3.0
wdm_3keV_s8 = theory.theory_P_lyas(params)

params['fast-model'] = '4keV_s8'
params['gadget-model'] = '4keV_s8'
params['m_wdm'] = 4.0
wdm_4keV_s8 = theory.theory_P_lyas(params)

params['fast-model'] = '6keV_s8'
params['gadget-model'] = '6keV_s8'
params['m_wdm'] = 6.0
wdm_6keV_s8 = theory.theory_P_lyas(params)

params['fast-model'] = '9keV_s8'
params['gadget-model'] = '9keV_s8'
params['m_wdm'] = 9.0
wdm_9keV_s8 = theory.theory_P_lyas(params)

params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.infty
cdm_s8 = theory.theory_P_lyas(params)

params['sigma8'] = 0.8659

params['fast-model'] = '3keV_splus'
params['gadget-model'] = '3keV_splus'
params['m_wdm'] = 3.0
wdm_3keV_splus = theory.theory_P_lyas(params)

params['fast-model'] = '4keV_splus'
params['gadget-model'] = '4keV_splus'
params['m_wdm'] = 4.0
wdm_4keV_splus = theory.theory_P_lyas(params)

params['fast-model'] = '6keV_splus'
params['gadget-model'] = '6keV_splus'
params['m_wdm'] = 6.0
wdm_6keV_splus = theory.theory_P_lyas(params)

params['fast-model'] = '9keV_splus'
params['gadget-model'] = '9keV_splus'
params['m_wdm'] = 9.0
wdm_9keV_splus = theory.theory_P_lyas(params)

params['fast-model'] = 'cdm_splus'
params['gadget-model'] = 'cdm_splus'
params['m_wdm'] = np.infty
cdm_splus = theory.theory_P_lyas(params)

params['sigma8'] = 0.7659

params['fast-model'] = '3keV_sminus'
params['gadget-model'] = '3keV_sminus'
params['m_wdm'] = 3.0
wdm_3keV_sminus = theory.theory_P_lyas(params)

params['fast-model'] = '4keV_sminus'
params['gadget-model'] = '4keV_sminus'
params['m_wdm'] = 4.0
wdm_4keV_sminus = theory.theory_P_lyas(params)

params['fast-model'] = '6keV_sminus'
params['gadget-model'] = '6keV_sminus'
params['m_wdm'] = 6.0
wdm_6keV_sminus = theory.theory_P_lyas(params)

params['fast-model'] = '9keV_sminus'
params['gadget-model'] = '9keV_sminus'
params['m_wdm'] = 9.0
wdm_9keV_sminus = theory.theory_P_lyas(params)

params['fast-model'] = 'cdm_sminus'
params['gadget-model'] = 'cdm_sminus'
params['m_wdm'] = np.infty
cdm_sminus = theory.theory_P_lyas(params)

# add zeta
params['sigma8'] = 0.7733
params['fast-model'] = 'zeta_p1'
params['gadget-model'] = 'zeta_p1'
params['m_wdm'] = 1./0.0933
zeta_p1 = theory.theory_P_lyas(params)

params['sigma8'] = 0.8331
params['fast-model'] = 'zeta_p2'
params['gadget-model'] = 'zeta_p2'
params['m_wdm'] = 1./0.13715
zeta_p2 = theory.theory_P_lyas(params)

params['sigma8'] = 0.7961
params['fast-model'] = 'zeta_p3'
params['gadget-model'] = 'zeta_p3'
params['m_wdm'] = 1./0.29557
zeta_p3 = theory.theory_P_lyas(params)

params['sigma8'] = 0.8615
params['fast-model'] = 'zeta_p4'
params['gadget-model'] = 'zeta_p4'
params['m_wdm'] = 1./0.16468
zeta_p4 = theory.theory_P_lyas(params)

params['sigma8'] = 0.8444
params['fast-model'] = 'zeta_p5'
params['gadget-model'] = 'zeta_p5'
params['m_wdm'] = 1./0.16847
zeta_p5 = theory.theory_P_lyas(params)

# fiducial model
params['sigma8'] = 0.8159
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.infty
ref = obs.observed_3D(params)

if next_gen > 0:
    ref.area_ddeg2 = 28000.0

# from wavelength range to z_bin
def obs_z(l):
        l_mean = np.sqrt(l * (l + 200.0)) 
        z = l_mean / 1215.67 - 1.0
        return z

# wavelength list
lmin_list = [3501.0 + i * 200.0 for i in range(13)]

# bins to do summation
z_bin = [obs_z(l) for l in lmin_list]
k_bin = np.linspace(0.06, 0.45, 40)
mu_bin = [0.1, 0.3, 0.5, 0.7, 0.9]


# calculate the bins by previous theoretical model
bins = np.zeros((len(z_bin)*len(k_bin)*len(mu_bin), 20))
i = 0
for z in z_bin:
    for k in k_bin:
        for mu in mu_bin:
            bins[i,0] = wdm_3keV_sminus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_3keV_sminus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,1] = wdm_4keV_sminus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_4keV_sminus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,2] = wdm_6keV_sminus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_6keV_sminus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,3] = wdm_9keV_sminus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_9keV_sminus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,4] = cdm_sminus.LyaLya_base_Mpc_norm(z, k, mu) + cdm_sminus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,5] = wdm_3keV_s8.LyaLya_base_Mpc_norm(z, k, mu) + wdm_3keV_s8.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,6] = wdm_4keV_s8.LyaLya_base_Mpc_norm(z, k, mu) + wdm_4keV_s8.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,7] = wdm_6keV_s8.LyaLya_base_Mpc_norm(z, k, mu) + wdm_6keV_s8.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,8] = wdm_9keV_s8.LyaLya_base_Mpc_norm(z, k, mu) + wdm_9keV_s8.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,9] = cdm_s8.LyaLya_base_Mpc_norm(z, k, mu) + cdm_s8.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,10] = wdm_3keV_splus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_3keV_splus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,11] = wdm_4keV_splus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_4keV_splus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,12] = wdm_6keV_splus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_6keV_splus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,13] = wdm_9keV_splus.LyaLya_base_Mpc_norm(z, k, mu) + wdm_9keV_splus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,14] = cdm_splus.LyaLya_base_Mpc_norm(z, k, mu) + cdm_splus.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,15] = zeta_p1.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p1.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,16] = zeta_p2.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p2.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,17] = zeta_p3.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p3.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,18] = zeta_p4.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p4.LyaLya_reio_Mpc_norm(z, k, mu)
            bins[i,19] = zeta_p5.LyaLya_base_Mpc_norm(z, k, mu) + zeta_p5.LyaLya_reio_Mpc_norm(z, k, mu)
            i += 1

# !! we use 1/m and sigma8 as parameter !!
inverse_mass = [1./3., 1./4., 1./6., 1./9., 0.]
sigma8 = [0.7659, 0.8159, 0.8659]

coords = np.zeros((20,3))

for i in range(3):
    for j in range(5):
        coords[5*i+j]= [inverse_mass[j], sigma8[i], 30.]

coords[15] = [0.09330, 0.7733, 29.6092]
coords[16] = [0.13715, 0.8331, 35.4506]
coords[17] = [0.29557, 0.7961, 25.1366]
coords[18] = [0.16468, 0.8615, 38.4571]
coords[19] = [0.16847, 0.8444, 34.4432]

# time to interpolate!
bins_inter = []

for i in range(len(bins)):
    bins_inter.append(interpolate.LinearNDInterpolator(coords, bins[i]))

end1 = time.time()
interp_time = end1 - start
print("Interpolation took {0:.1f} seconds".format(interp_time))

# calculate bins of cdm reference model and variance
ref_bin = []
var_bin = []
for i in range(len(z_bin)):
    z = z_bin[i]
    ref.lmin = lmin_list[i]
    ref.lmax = lmin_list[i] + 200.0
    peff, pw, pn = ref.EffectiveDensityAndNoise()
    if next_gen > 0:
        pw /= 3.
        pn /= 3.
    # for each redshift, we calculate Pw2D and PN_eff once to save some time
    #dkms_dmpc = ref.convert.dkms_dMpc(z)          # convert to funny units
    #dmpc_ddeg = ref.convert.dMpc_ddeg(z)
    for k in k_bin:
        for mu in mu_bin:
            #kp_kms = k * mu / dkms_dmpc
            #kt_deg = k * np.sqrt(1.0 - mu**2) * dmpc_ddeg
            ref_bin.append(cdm_s8.LyaLya_base_Mpc_norm(z, k, mu) + cdm_s8.LyaLya_reio_Mpc_norm(z, k, mu))
            var_bin.append(ref.VarFluxP3D_Mpc_yao(k, mu, 0.01, 0.2, Pw2D=pw, PN_eff=pn))    # Yao: note that we use linear k bins here, not log k bins, so the calculation of Nmode need to be changed in observed_3D.py


end2 = time.time()
ref_time = end2 - end1
print("Reference preparation took {0:.1f} seconds".format(ref_time))

# the time taken for each likelihood calculation
cal_time = []
#  log-probability function

def log_prob(theta, ref, var):
    cal_start = time.time()
    inver_mass, sigma, zeta = theta
    test = (bins_inter[1])(inver_mass,sigma,zeta)
    if (np.isnan(test)):
        return -np.inf, -np.inf
    else:
        log_p = 0
        for i in range(len(bins_inter)):
            log_p += ((bins_inter[i])(inver_mass,sigma,zeta) - ref[i])**2 / var[i]
        cal_end = time.time()
        cal_time.append(cal_end - cal_start)
        if (np.isnan(log_p)):
            return -np.inf, -np.inf
        return -0.5 * log_p, 0.0

# gaussian prior

'''
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
        # use gaussian prior for sigma8 from PLANCK 2015
        prior = -0.5*(sigma-0.8159)**2 / 0.0086**2
        return -0.5 * log_p + prior, prior
'''

nw = 32
nd = 3

# we need to make the initial value in the prior range
initial = np.zeros((nw, nd))
for l in range(nw):
    initial[l,0] = np.random.rand()/3
    initial[l,1] = 0.7659 + np.random.rand() * 0.1
    initial[l,2] = 25.2 + np.random.rand() * (38.-25.2)
print(initial)

# run mcmc chain
filename = "chain_lya_NGen%d_zeta_f.h5"%next_gen
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, args=(ref_bin, var_bin), backend=backend, moves=emcee.moves.StretchMove(a=4.0))
sampler.run_mcmc(initial, 50000, progress=False)

end3 = time.time()
mcmc_time = end3 - end2
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
fig.savefig('chain_lya_NGen%d_zeta_f.pdf'%next_gen)


samples = sampler.get_chain(flat=True, discard=2000, thin=20)
fig1 = corner.corner(
    samples, labels=['1keV / m', r'$\sigma_8$', r'$\zeta$'], truths=[0,0.8159,30.])
fig1.savefig('corner_lya_NGen%d_zeta_f.pdf'%next_gen)

cal_time = np.array(cal_time, dtype=float)
print('calculation time for each step: ', np.mean(cal_time))
print("length: ", len(cal_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
