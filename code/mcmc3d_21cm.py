import numpy as np
import theory_P_21cm as theory
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

"""
    3-parameter MCMC forecast for skalow and puma using 21 cm IM power spectrum.
    Only use realization 1.
    3 parameter: 1/m_WDM, sigma8, zeta
    input: [telescope]: skalow or puma
"""

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
params['alphas'] = -0.002
params['taure'] = 0.066
params['bHI'] = 2.82
params['OHI'] = 1.18e-3 * 1.e3
params['fast-realization'] = 'r1'
params['gadget-realization'] = 'r1'
params['band'] = 'g'
params['telescope'] = tele
params['beam'] = 32 # bandwidth in MHz
params['z_max_pk'] = 5.5 # only farmer would have 35 for running patchy class 
params['P_k_max_1/Mpc'] = 10
params['pickle'] = False # only farmer would have True here

if params['telescope'] == 'skalow':
    D_dish = 40.0 # meter
    params['t_int'] = 5000. # hours
    z_bin = [3.65, 3.95, 4.25, 4.55, 4.85, 5.15, 5.45]
    dz = 0.3
elif params['telescope'] == 'puma':
    D_dish = 6.0
    params['t_int'] = 1000.
    z_bin = [3.6+0.2*i for i in range(10)]
    dz = 0.2
else:
    print("Cannot determine the telescope!")
    exit(1)

# Yao: need to check the names of pickles and files

# prepare theoretical model for interpolation
params['sigma8'] = 0.8159

params['fast-model'] = '3keV_s8'
params['gadget-model'] = '3keV_s8'
params['m_wdm'] = 3.0
wdm_3keV_s8 = theory.theory_P_21(params)

params['fast-model'] = '4keV_s8'
params['gadget-model'] = '4keV_s8'
params['m_wdm'] = 4.0
wdm_4keV_s8 = theory.theory_P_21(params)

params['fast-model'] = '6keV_s8'
params['gadget-model'] = '6keV_s8'
params['m_wdm'] = 6.0
wdm_6keV_s8 = theory.theory_P_21(params)

params['fast-model'] = '9keV_s8'
params['gadget-model'] = '9keV_s8'
params['m_wdm'] = 9.0
wdm_9keV_s8 = theory.theory_P_21(params)

params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.infty
cdm_s8 = theory.theory_P_21(params)

params['sigma8'] = 0.8659

params['fast-model'] = '3keV_splus'
params['gadget-model'] = '3keV_splus'
params['m_wdm'] = 3.0
wdm_3keV_splus = theory.theory_P_21(params)

params['fast-model'] = '4keV_splus'
params['gadget-model'] = '4keV_splus'
params['m_wdm'] = 4.0
wdm_4keV_splus = theory.theory_P_21(params)

params['fast-model'] = '6keV_splus'
params['gadget-model'] = '6keV_splus'
params['m_wdm'] = 6.0
wdm_6keV_splus = theory.theory_P_21(params)

params['fast-model'] = '9keV_splus'
params['gadget-model'] = '9keV_splus'
params['m_wdm'] = 9.0
wdm_9keV_splus = theory.theory_P_21(params)

params['fast-model'] = 'cdm_splus'
params['gadget-model'] = 'cdm_splus'
params['m_wdm'] = np.infty
cdm_splus = theory.theory_P_21(params)

params['sigma8'] = 0.7659

params['fast-model'] = '3keV_sminus'
params['gadget-model'] = '3keV_sminus'
params['m_wdm'] = 3.0
wdm_3keV_sminus = theory.theory_P_21(params)

params['fast-model'] = '4keV_sminus'
params['gadget-model'] = '4keV_sminus'
params['m_wdm'] = 4.0
wdm_4keV_sminus = theory.theory_P_21(params)

params['fast-model'] = '6keV_sminus'
params['gadget-model'] = '6keV_sminus'
params['m_wdm'] = 6.0
wdm_6keV_sminus = theory.theory_P_21(params)

params['fast-model'] = '9keV_sminus'
params['gadget-model'] = '9keV_sminus'
params['m_wdm'] = 9.0
wdm_9keV_sminus = theory.theory_P_21(params)

params['fast-model'] = 'cdm_sminus'
params['gadget-model'] = 'cdm_sminus'
params['m_wdm'] = np.infty
cdm_sminus = theory.theory_P_21(params)

# add zeta

params['m_wdm'] = np.infty

params['sigma8'] = 0.7659
params['fast-model'] = 'zeta_p1'
params['gadget-model'] = 'zeta_p1'
zeta_p1 = theory.theory_P_21(params)

params['fast-model'] = 'zeta_p2'
params['gadget-model'] = 'zeta_p2'
zeta_p2 = theory.theory_P_21(params)

params['sigma8'] = 0.8659
params['fast-model'] = 'zeta_p3'
params['gadget-model'] = 'zeta_p3'
zeta_p3 = theory.theory_P_21(params)

params['fast-model'] = 'zeta_p4'
params['gadget-model'] = 'zeta_p4'
zeta_p4 = theory.theory_P_21(params)

params['m_wdm'] = 3.0

params['sigma8'] = 0.7659
params['fast-model'] = 'zeta_p5'
params['gadget-model'] = 'zeta_p5'
zeta_p5 = theory.theory_P_21(params)

params['fast-model'] = 'zeta_p6'
params['gadget-model'] = 'zeta_p6'
zeta_p6 = theory.theory_P_21(params)

params['sigma8'] = 0.8659
params['fast-model'] = 'zeta_p7'
params['gadget-model'] = 'zeta_p7'
zeta_p7 = theory.theory_P_21(params)

params['fast-model'] = 'zeta_p8'
params['gadget-model'] = 'zeta_p8'
zeta_p8 = theory.theory_P_21(params)


# !! we use 1/m, sigma8 and zeta as parameters !!
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


bins_inter = []
ref_bin = []
var_bin = []


bin_class = wedge.bins(dish_D=D_dish)
for z in z_bin:
    bin_class.z = z
    bin_class.dz = dz
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
            one_bin[0] = wdm_3keV_sminus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[1] = wdm_4keV_sminus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[2] = wdm_6keV_sminus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[3] = wdm_9keV_sminus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[4] = cdm_sminus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[5] = wdm_3keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[6] = wdm_4keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[7] = wdm_6keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[8] = wdm_9keV_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[9] = cdm_s8.P3D_21_Mpc_norm(z, k, mu)
            one_bin[10] = wdm_3keV_splus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[11] = wdm_4keV_splus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[12] = wdm_6keV_splus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[13] = wdm_9keV_splus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[14] = cdm_splus.P3D_21_Mpc_norm(z, k, mu)
            one_bin[15] = zeta_p1.P3D_21_Mpc_norm(z, k, mu)
            one_bin[16] = zeta_p2.P3D_21_Mpc_norm(z, k, mu)
            one_bin[17] = zeta_p3.P3D_21_Mpc_norm(z, k, mu)
            one_bin[18] = zeta_p4.P3D_21_Mpc_norm(z, k, mu)
            one_bin[19] = zeta_p5.P3D_21_Mpc_norm(z, k, mu)
            one_bin[20] = zeta_p6.P3D_21_Mpc_norm(z, k, mu)
            one_bin[21] = zeta_p7.P3D_21_Mpc_norm(z, k, mu)
            one_bin[22] = zeta_p8.P3D_21_Mpc_norm(z, k, mu)
            bins_inter.append(interpolate.LinearNDInterpolator(coords, one_bin))
            ref_bin.append(cdm_s8.P3D_21_Mpc_norm(z, k, mu))
            var_bin.append(cdm_s8.Var_autoHI_Mpc_yao(z, k, mu, dz, dk, dmu)) # Yao: note this func assume dmu=0.2, which might need modification



end1 = time.time()
interp_time = end1 - start
print("Preparing models took {0:.1f} seconds".format(interp_time))


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
    

nw = 32
nd = 3

# we need to generate the initial value in the prior range
initial = np.zeros((nw, nd))
for l in range(nw):
    initial[l,0] = np.random.rand()/3.
    initial[l,1] = 0.7659 + np.random.rand() * 0.1
    initial[l,2] = 20.0 + np.random.rand() * 20.0


# run mcmc chain
filename = "chain_21cm_%s_zeta_pop2.h5"%(tele)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, args=(ref_bin, var_bin), backend=backend, moves=emcee.moves.StretchMove(a=4.0))
sampler.run_mcmc(initial, 50000, progress=False)

end2 = time.time()
mcmc_time = end2 - end1
print("MCMC took {0:.1f} seconds".format(mcmc_time))

cal_time = np.array(cal_time, dtype=float)
print('calculation time for each step: ', np.mean(cal_time))
print("length: ", len(cal_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))

