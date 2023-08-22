import numpy as np
import theory_P_cross_full as theory
import wedge
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
import sys

os.environ["OMP_NUM_THREADS"] = "1"

start = time.time()

tele = sys.argv[1]

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
params['fast-realization'] = 'r1'
params['gadget-realization'] = 'r1'
params['m_wdm'] = np.infty # for cdm is np.infty
params['band'] = 'g'
params['sigma8'] = 0.8159
params['telescope'] = tele
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
wdm_3keV_s8 = theory.theory_P_cross(params)

params['fast-model'] = '4keV_s8'
params['gadget-model'] = '4keV_s8'
params['m_wdm'] = 4.0
wdm_4keV_s8 = theory.theory_P_cross(params)

params['fast-model'] = '6keV_s8'
params['gadget-model'] = '6keV_s8'
params['m_wdm'] = 6.0
wdm_6keV_s8 = theory.theory_P_cross(params)

params['fast-model'] = '9keV_s8'
params['gadget-model'] = '9keV_s8'
params['m_wdm'] = 9.0
wdm_9keV_s8 = theory.theory_P_cross(params)

params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.infty
cdm_s8 = theory.theory_P_cross(params)

# bins to do summation
if params['telescope'] == 'skalow':
    D_dish = 40.0 # meter
elif params['telescope'] == 'puma':
    D_dish = 6.0
else:
    print("Cannot determine the telescope!")
    exit(1)

# !! we use 1/m and sigma8 as parameter !!
inverse_mass = [1/3, 1/4, 1/6, 1/9, 0]

z_bin = [3.5+0.2*i for i in range(11)]

bins_inter = []
ref_bin = []
var_bin = []


bin_class = wedge.bins(dish_D=D_dish)
for z in z_bin:
    bin_class.z = z
    k_bin, dk_bin, kmin = bin_class.k_bins()
    k_parallel_min = bin_class.k_parallel_min()
    k_perp_min = bin_class.k_perp_min()
    mu_bin, dmu = bin_class.mu_bins()
    mu_wedge = bin_class.mu_wedge()
    for k, dk in zip(k_bin, dk_bin):
        for mu in mu_bin:
            k_parallel = k * mu
            k_perp = k * np.sqrt(1-mu**2)
            if (k_parallel<k_parallel_min) or (k_perp<k_perp_min) or mu<mu_wedge:
                continue
            print("%f  %f  %f  %f  %f"%(z, k, dk, mu, dmu))
            one_bin = np.zeros(5)
            one_bin[0] = wdm_3keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[1] = wdm_4keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[2] = wdm_6keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[3] = wdm_9keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[4] = cdm_s8.P3D_21_Mpc_norm(z, k, mu)
            bins_inter.append(interpolate.interp1d(inverse_mass, one_bin))
            ref_bin.append(cdm_s8.P3D_21_Mpc_norm(z, k, mu))
            var_bin.append(cdm_s8.Var_autoHI_Mpc(z, k, mu, dk, dmu)) # Yao: note this func assume dmu=0.2, which might need modification



end1 = time.time()
interp_time = end1 - start
print("Preparing models took {0:.1f} seconds".format(interp_time))


# the time taken for each likelihood calculation
cal_time = []
#  log-probability function
def log_prob(inver_mass, ref, var):
    cal_start = time.time()
    if inver_mass > 1/3 or inver_mass < 0:
        return -np.inf, -np.inf
    else:
        log_p = 0
        for i in range(len(bins_inter)):
            log_p += ((bins_inter[i])(inver_mass) - ref[i])**2 / var[i]
        cal_end = time.time()
        cal_time.append(cal_end - cal_start)
        return -0.5 * log_p, 0.0

nw = 32
nd = 1

# we need to make the initial value in the prior range
initial = np.zeros((nw, nd))
for l in range(nw):
    initial[l,0] = np.random.rand()/3

# run mcmc chain
filename = "chain1d_21cm_%s.h5"%tele
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, args=(ref_bin, var_bin), backend=backend)
sampler.run_mcmc(initial, 20000, progress=False)

end2 = time.time()
mcmc_time = end2 - end1
print("MCMC took {0:.1f} seconds".format(mcmc_time))

fig, axs = plt.subplots(1)
samples1 = sampler.get_chain()
axs.plot(range(len(samples1)), samples1[:, :, 0], "k", alpha=0.3)
axs.set_xlabel('step number')
axs.set_ylabel('1keV / m')
fig.savefig('chain1d_21cm_%s.pdf'%tele)


samples = sampler.get_chain(flat=True, discard=500, thin=20)
fig1 = corner.corner(
    samples, labels=['1keV / m'], truths=[0])
fig1.savefig('corner1d_21cm_%s.pdf'%tele)

cal_time = np.array(cal_time, dtype=float)
print('calculation time for each step: ', np.mean(cal_time))
print("length: ", len(cal_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
