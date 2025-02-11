import numpy as np
import pickle
import sys

"""
    Look! A pickle farmer?
    
    We run this script once per model.

    inputs: model fast_realization gadget_realization m_wdm(in keV or 'cdm') sigma8

    outputs:
    psi_[gadget_realization]_[model].pkl
    pat_[fast_realization]_[model].pkl
    p_mXi_[fast_realization]_[model]_[gadget_realization]_[model].pkl
    p_mpsi_[fast_realization]_[model]_[gadget_realization]_[model].pkl
    
"""

""" FIRST: carefully tune the parameters of the dictionary to your model """

# put attention to sigma8 value, m_wdm value

# after that put attention to the fast-model and fast-realization

# finally, adjust the gadget-realization and gadget-model

model = sys.argv[1]
fast_realization = sys.argv[2]
gadget_realization = sys.argv[3]


# dictionary
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
params['fast-model'] = model
params['fast-realization'] = fast_realization
params['gadget-realization'] = gadget_realization
params['gadget-model'] = model # so it should be something like 9keV_s8 now
params['band'] = 'g'
params['telescope'] = 'skalow'
params['t_int'] = 5000
params['beam'] = 32 # think about this one
params['z_max_pk'] = 35 # 35 for running patchy class and 5.5 for everything else
params['P_k_max_1/Mpc'] = 110
params['pickle'] = True # this needs to be True for this script.

if sys.argv[4]=='cdm':
    params['m_wdm'] = np.inf
else:
    params['m_wdm'] = float(sys.argv[4])

params['sigma8'] = float(sys.argv[5])


print('\n')
print('***********************************************************')
print('This is your dictionary, please check the sigma8, m_WDM values and others!!!')
print('***********************************************************')
print(params)
print('\n')
# let's import all the important classes and do their pickles one by one

""" Let's do the transparency """
print('***********************************************************')
print('Farmer is producing psi pickles, YUM')

import psi

psi_veggie = psi.transparency(params)
print('\n')

print('***********************************************************')
print('Farmer is producing patchy bubble pickles, YUM')

import patchy

pat = patchy.P_bubble(params)

pat.pickle_this()
print('\n')


print('***********************************************************')
print('Farmer is producing 21 cm pickles, YUM')
import patchy_reion_21 as p21

p_21 = p21.P_21_obs(params)

p_21.pickle_this()
print('\n')
print('***********************************************************')
print('Farmer is producing Lya pickles, YUM')

import theory_P_lyas_arinyo as plya

p_lya = plya.theory_P_lyas(params)

p_lya.pickle_this()
print('\n')
print('***********************************************************')
print('Farmer is done for today, here is a summary of your model')
print('\n')

print('***********************************************************')
print('21cmfast realization: ', params['fast-realization'])
print('21cmfast model: ', params['fast-model'])
print('Gadget realization: ', params['gadget-realization'])
print('Gadget model: ', params['gadget-model'])
print('AND a WDM mass [in keV] of ', params['m_wdm'])
print('AND a sigma8 of ', params['sigma8'])
