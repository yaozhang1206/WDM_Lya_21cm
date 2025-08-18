import numpy as np
import theory_P_21cm as theory
import wedge
import emcee
from matplotlib import pyplot as plt
from scipy import interpolate
import time
import os
import sys


"""
    4-parameter MCMC forecast for skalow and puma using 21 cm IM power spectrum.
    Only use realization 1.
    4 parameter: 1/m_WDM, sigma8, zeta, M_turn
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
params['mnu'] = 0.194
params['ns'] = 0.9667
params['As'] = 2.142 # 10^9 * As
params['alphas'] = -0.002
params['taure'] = 0.066
params['bHI'] = 2.82
params['OHI'] = 1.18e-3 * 1.e3
params['fast-realization'] = 'r1'   # yao: need to be fixed later
params['gadget-realization'] = 'r1'
params['band'] = 'g'
params['telescope'] = tele
params['beam'] = 32 # think about this one
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

# fiducial model
params['sigma8'] = 0.8159
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.inf
cdm_s8 = theory.theory_P_21(params)


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
models = []
coords = []
for i,dm in enumerate(dm_model):
    for j,s8 in enumerate(s8_model):
        name = "%s_%s"%(dm,s8)
        params['fast-model'] = name
        params['gadget-model'] = name
        params['m_wdm'] = dm_mass[i]
        params['sigma8'] = s8s[j]
        models.append(theory.theory_P_21(params))
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
            models.append(theory.theory_P_21(params))
            coords.append([inverse_mass_short[i],s8s_short[j],zetas[k],8.7])
            for l,mturn in enumerate(mturn_model):
                name1 = "%s_%s_%s_%s"%(dm,s8,zeta,mturn)
                params['fast-model'] = name1
                models.append(theory.theory_P_21(params))
                coords.append([inverse_mass_short[i],s8s_short[j],zetas[k],mturns[l]])

coords = np.array(coords)
model_num = len(models)

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
            one_bin = np.zeros(model_num)
            for i in range(model_num):
                one_bin[i] = models[i].P3D_21_Mpc_norm(z, k, mu)
            bins_inter.append(interpolate.LinearNDInterpolator(coords, one_bin))
            ref_bin.append(cdm_s8.P3D_21_Mpc_norm(z, k, mu))
            var_bin.append(cdm_s8.Var_autoHI_Mpc_yao(z, k, mu, dz, dk, dmu)) # Yao: note this func assume dmu=0.2, which might need modification

end1 = time.time()
interp_time = end1 - start
print("Preparing models took {0:.1f} seconds".format(interp_time))


#  log-probability function

def log_prob(theta, ref, var):
    in_mass, sigma, zeta, mturn = theta
    test = (bins_inter[1])(in_mass,sigma,zeta,mturn)
    if (np.isnan(test)):
        return -np.inf, -np.inf
    else:
        log_p = 0
        for i in range(len(bins_inter)):
            log_p += ((bins_inter[i])(in_mass,sigma,zeta,mturn) - ref[i])**2 / var[i]
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


# run mcmc chain
filename = "4d/chain_21cm_%s_4d.h5"%(tele)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, args=(ref_bin, var_bin), backend=backend, moves=emcee.moves.StretchMove(a=4.0))
sampler.run_mcmc(initial, 50000, progress=False)

end2 = time.time()
mcmc_time = end2 - end1
print("MCMC took {0:.1f} seconds".format(mcmc_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))

