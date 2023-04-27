# WDM_Lyman_alpha & WDM_21cm

Code requirements: classy (which also requires cython)

Code architecture is now in "proper pipeline":
theory_P_lyas_arinyo.py is a python class in charge of building the functions that we need to compute the memory of reionization in the lyman-alpha forest. Its main contribution is the ability to pickle the memory of reionization for later use, P_m,psi (z_obs,k) -> e.g. p_mpsi_ave_cdm_ave_cdm.pkl

patchy.py is a python class whose objective is to construct a pickle of P_m,xHI (z_re,k) -> e.g. pat_r1_cdm.pkl

psi.py class that stores the IGM transparency results psi (z_re, z_obs) -> e.g. psi_r4_9keV.pkl

Pm_DM.py is a class in charge of computing the matter power spectrum. 

patchy_reion_21.py is the python class in charge of the memory of reionization in 21cm, i.e. P_m,Xi (z_re, k) -> p_mXi_r2_3keV_r2_3keV.pkl

farmer.py is a script that will generate all the pickles for a given model. Only needs to be run once per model. 

theory_P_cross_full.py is the master class, it controls the variance and construct the observables.


master is a jupyter notebook which we can use to run our analysis. (old) 
