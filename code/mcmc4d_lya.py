import numpy as np
import theory_P_lyas_arinyo as theory
import observed_3D as obs
import emcee
from scipy import interpolate
import time
import os
import sys

"""
    4-parameter MCMC forecast for DESI and Stage V using lyman alpha forest power spectrum.
    Only use realization 1.
    4 parameters: 1/m_WDM, sigma8, zeta, M_turn
    input: [next_gen]: 0 or 1
    0: DESI; 1: Stage V
"""


os.environ["OMP_NUM_THREADS"] = "1"

start = time.time()

next_gen = int(sys.argv[1])

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
params['fast-realization'] = 'r1'   # yao: need to be fixed later
params['gadget-realization'] = 'r1'
params['band'] = 'g'
params['telescope'] = 'skalow'
params['t_int'] = 1000
params['beam'] = 32 # think about this one
params['z_max_pk'] = 5.5 # only farmer would have 35 for running patchy class 
params['P_k_max_1/Mpc'] = 10
params['pickle'] = False # only farmer would have True here

# fiducial model
params['sigma8'] = 0.8159
params['fast-model'] = 'cdm_s8'
params['gadget-model'] = 'cdm_s8'
params['m_wdm'] = np.inf
ref = obs.observed_3D(params)
cdm_s8 = theory.theory_P_lyas(params)

if next_gen > 0:
    ref.area_ddeg2 = 28000.0

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
        models.append(theory.theory_P_lyas(params))
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
            print(name1)
            models.append(theory.theory_P_lyas(params))
            coords.append([inverse_mass_short[i],s8s_short[j],zetas[k],8.7])
            for l,mturn in enumerate(mturn_model):
                name1 = "%s_%s_%s_%s"%(dm,s8,zeta,mturn)
                params['fast-model'] = name1
                print(name1)
                models.append(theory.theory_P_lyas(params))
                coords.append([inverse_mass_short[i],s8s_short[j],zetas[k],mturns[l]])

coords = np.array(coords)
model_num = len(models)

# from wavelength range to z_bin
def obs_z(lmin, lmax):
        l_mean = np.sqrt(lmin * lmax) 
        z = l_mean / 1215.67 - 1.0
        return z

k_bin = np.linspace(0.06, 0.35, 30)
mu_bin = [0.125,0.375,0.625,0.875]
# wavelength list
lmin_list = [3.*1215.67, 4.*1215.67]
lmax_list = [4.*1215.67, 5.*1215.67]
# bins to do summation
z_bin = [obs_z(lmin_list[i],lmax_list[i]) for i in range(2)]
bin_num = len(z_bin)*len(k_bin)*len(mu_bin)
bins = np.zeros((bin_num,model_num))
for i in range(model_num):
    j = 0
    for z in z_bin:
        for k in k_bin:
            for mu in mu_bin:
                bins[j,i] = models[i].LyaLya_base_Mpc_norm(z, k, mu) + models[i].LyaLya_reio_Mpc_norm(z, k, mu)
                j += 1

# time to interpolate!
bins_inter = []

for i in range(bin_num):
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
    ref.lmax = lmax_list[i]
    peff, pw, pn = ref.EffectiveDensityAndNoise()
    if next_gen > 0:
        pw /= 3.
        pn /= 3.
    for k in k_bin:
        for mu in mu_bin:
            ref_bin.append(cdm_s8.LyaLya_base_Mpc_norm(z, k, mu) + cdm_s8.LyaLya_reio_Mpc_norm(z, k, mu))
            var_bin.append(ref.VarFluxP3D_Mpc_yao(k, mu, 0.01, 0.25, Pw2D=pw, PN_eff=pn))    # Yao: note that we use linear k bins here, not log k bins, so the calculation of Nmode need to be changed in observed_3D.py


end2 = time.time()
ref_time = end2 - end1
print("Reference preparation took {0:.1f} seconds".format(ref_time))

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
print(initial)

# run mcmc chain
filename = "chain_lya_NGen%d_4d.h5"%(next_gen)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers=nw, ndim=nd)
sampler = emcee.EnsembleSampler(nwalkers = nw, ndim = nd, log_prob_fn = log_prob, args=(ref_bin, var_bin), backend=backend, moves=emcee.moves.StretchMove(a=4.0))
sampler.run_mcmc(initial, 50000, progress=False)

end3 = time.time()
mcmc_time = end3 - end2
print("MCMC took {0:.1f} seconds".format(mcmc_time))

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))


