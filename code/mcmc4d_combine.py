import numpy as np
import theory_P_cross_full as theory_21
import theory_P_lyas_arinyo as theory_lya
import observed_3D as obs
import wedge
import emcee
from matplotlib import pyplot as plt
from scipy import interpolate
import time
import os
import sys

"""
    4-parameter MCMC forecast for DESI+SKA1-LOW and PUMA+Stage V using lyman alpha forest and 21 cm IM power spectra.
    Only use realization 1.
    4 parameter: 1/m_WDM, sigma8, zeta, mturn
    input: [telescope]: skalow or puma
    skalow: DESI+SKA1-LOW; puma: PUMA+Stage V
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
params['As'] = 2.142 # 10^9 * As
params['mnu'] = 0.194
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
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.inf
cdm_s8_lya = theory_lya.theory_P_lyas(params)
cdm_s8_21 = theory_21.theory_P_cross(params)
ref_lya = obs.observed_3D(params)
if tele == 'puma':
    ref_lya.area_ddeg2 = 28000.


# dark matter models
dm_model = ['cdm','9keV','6keV','4keV','3keV']
inverse_mass = [0., 1./9., 1./6., 1./4., 1./3.]
dm_mass = [np.inf, 9., 6., 4., 3.]
dm_model_short = ['cdm','3keV']
inverse_mass_short = [0., 1./3.]
dm_mass_short = [np.inf, 3.]
#sigma8
s8_model = ['sminus','s8','splus']
s8s = [0.7659,0.8159,0.8659]
s8_model_short = ['sminus','splus']
s8s_short = [0.7659,0.8659]
# zeta
zeta_model = ['zminus','zplus']
zetas = [20,35]
# M_turn
mturn_model = ['mlow','mhigh']
mturns = [8., 9.0]


# prepare theoretical model for interpolation
models_lya = []
models_21 = []
coords = []
for i,dm in enumerate(dm_model):
    for j,s8 in enumerate(s8_model):
        name = "%s_%s"%(dm,s8)
        params['fast-model'] = name
        params['gadget-model'] = name
        params['m_wdm'] = dm_mass[i]
        params['sigma8'] = s8s[j]
        models_lya.append(theory_lya.theory_P_lyas(params))
        models_21.append(theory_21.theory_P_cross(params))
        coords.append([inverse_mass[i],s8s[j],24,8.7])


for i,dm in enumerate(dm_model_short):
    for j,s8 in enumerate(s8_model_short):
        name2 = "%s_%s"%(dm,s8)       # zeta and Mturn are irrelevant to gadget
        params['gadget-model'] = name2
        params['m_wdm'] = dm_mass_short[i]
        params['sigma8'] = s8s_short[j]
        for k,zeta in enumerate(zeta_model):
            name1 = "%s_%s_%s"%(dm,s8,zeta)
            params['fast-model'] = name1
            models_lya.append(theory_lya.theory_P_lyas(params))
            models_21.append(theory_21.theory_P_cross(params))
            coords.append([inverse_mass_short[i],s8s_short[j],zetas[k],8.7])
            for l,mturn in enumerate(mturn_model):
                name1 = "%s_%s_%s_%s"%(dm,s8,zeta,mturn)
                params['fast-model'] = name1
                models_lya.append(theory_lya.theory_P_lyas(params))
                models_21.append(theory_21.theory_P_cross(params))
                coords.append([inverse_mass_short[i],s8s_short[j],zetas[k],mturns[l]])

coords = np.array(coords)
model_num = len(models_lya)


#*********** Lya forest *********#
# from wavelength range to z_bin
def obs_z(lmin, lmax):
        l_mean = np.sqrt(lmin * lmax) 
        z = l_mean / 1215.67 - 1.0
        return z

# wavelength list
lmin_list = [3.*1215.67, 4.*1215.67]
lmax_list = [4.*1215.67, 5.*1215.67]

# bins for lya
z_bin_lya = [obs_z(lmin_list[i],lmax_list[i]) for i in range(2)]
k_bin_lya = np.linspace(0.06, 0.35, 30)
mu_bin_lya = [0.125,0.375,0.625,0.875]


bins_lya = np.zeros((len(z_bin_lya)*len(k_bin_lya)*len(mu_bin_lya), model_num))
for i in range(model_num):
    j = 0
    for z in z_bin_lya:
        for k in k_bin_lya:
            for mu in mu_bin_lya:
                bins_lya[j,i] = models_lya[i].LyaLya_base_Mpc_norm(z, k, mu) + models_lya[i].LyaLya_reio_Mpc_norm(z, k, mu)
                j += 1

# time to interpolate!
bins_inter_lya = []

for i in range(len(bins_lya)):
    bins_inter_lya.append(interpolate.LinearNDInterpolator(coords, bins_lya[i]))

ref_bin_lya = []
var_bin_lya = []
for i in range(len(z_bin_lya)):
    z = z_bin_lya[i]
    ref_lya.lmin = lmin_list[i]
    ref_lya.lmax = lmax_list[i]
    peff, pw, pn = ref_lya.EffectiveDensityAndNoise() # for each redshift, we calculate Pw2D and PN_eff once to save some time
    # for puma, we assume it combines with DESI++ instruments, and P_N and P_w can be reduced by a factor of 3
    if tele == 'puma':
        pw /= 3.
        pn /= 3.
    for k in k_bin_lya:
        for mu in mu_bin_lya:
            ref_bin_lya.append(cdm_s8_lya.LyaLya_base_Mpc_norm(z, k, mu) + cdm_s8_lya.LyaLya_reio_Mpc_norm(z, k, mu))
            var_bin_lya.append(ref_lya.VarFluxP3D_Mpc_yao(k, mu, 0.01, 0.25, Pw2D=pw, PN_eff=pn))    # Yao: note that we use linear k bins here, not log k bins, so the calculation of Nmode need to be changed in observed_3D.py


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
            one_bin = np.zeros(model_num)
            for i in range(model_num):
                one_bin[i] = models_21[i].P3D_21_Mpc_norm(z, k, mu)
            bins_inter_21.append(interpolate.LinearNDInterpolator(coords, one_bin))
            ref_bin_21.append(cdm_s8_21.P3D_21_Mpc_norm(z, k, mu))
            var_bin_21.append(cdm_s8_21.Var_autoHI_Mpc_yao(z, k, mu, dz_21, dk, dmu)) # Yao: note this func assume dmu=0.2, which might need modification



end1 = time.time()
interp_time = end1 - start
print("Preparing models took {0:.1f} seconds".format(interp_time))

#  log-probability function
def log_prob(theta):
    in_mass, sigma, zeta, mturn = theta
    test = (bins_inter_lya[1])(in_mass,sigma,zeta,mturn)
    if (np.isnan(test)):
        return -np.inf, -np.inf
    else:
        log_p = 0
        for i in range(len(bins_inter_lya)):
            log_p += ((bins_inter_lya[i])(in_mass,sigma,zeta,mturn) - ref_bin_lya[i])**2 / var_bin_lya[i]

        for i in range(len(bins_inter_21)):
            log_p += ((bins_inter_21[i])(in_mass,sigma,zeta,mturn) - ref_bin_21[i])**2 / var_bin_21[i]
        if (np.isnan(log_p)):
            return -np.inf, -np.inf
        return -0.5 * log_p, 0.0


nw = 64
nd = 4

# we need to make the initial value in the prior range
initial = np.zeros((nw, nd))
for l in range(nw):
    initial[l,0] = np.random.rand()/3
    initial[l,1] = 0.7659 + np.random.rand() * 0.1
    initial[l,2] = 20. + np.random.rand() * 15.
    initial[l,3] = 8.0 + np.random.rand()
print(initial)

# run mcmc chain
filename = "4d/chain_combine_%s_4d.h5"%(tele)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, backend=backend, moves=emcee.moves.StretchMove(a=4.0))
sampler.run_mcmc(initial, 50000, progress=False)

end2 = time.time()
mcmc_time = end2 - end1
print("MCMC took {0:.1f} seconds".format(mcmc_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))

