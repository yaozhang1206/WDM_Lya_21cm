import numpy as np
import matplotlib
matplotlib.use("agg")
import theory_P_lyas_fur_yao as th_y
import matplotlib.pyplot as plt
# nomenclature to deal with this procedure model_21cmrealization_gadgetrealization, e.g. 
# r1_ave_3keV uses gadget mean.

# need to do average Gadget and all the 21cm fast realizations
r1_ave_cdm = th_y.theory_P_lyas(np.infty, '21cm_r1_cdm', 'gadget_ave8_cdm')
r2_ave_cdm = th_y.theory_P_lyas(np.infty, '21cm_r2_cdm', 'gadget_ave8_cdm')
r3_ave_cdm = th_y.theory_P_lyas(np.infty, '21cm_r3_cdm', 'gadget_ave8_cdm')
r4_ave_cdm = th_y.theory_P_lyas(np.infty, '21cm_r4_cdm', 'gadget_ave8_cdm')


r1_ave_3keV = th_y.theory_P_lyas(3.0, '21cm_r1_wdm3', 'gadget_ave8_3keV')
r2_ave_3keV = th_y.theory_P_lyas(3.0, '21cm_r2_wdm3', 'gadget_ave8_3keV')
r3_ave_3keV = th_y.theory_P_lyas(3.0, '21cm_r3_wdm3', 'gadget_ave8_3keV')
r4_ave_3keV = th_y.theory_P_lyas(3.0, '21cm_r4_wdm3', 'gadget_ave8_3keV')

r1_ave_4keV = th_y.theory_P_lyas(4.0, '21cm_r1_wdm4', 'gadget_ave8_4keV')
r2_ave_4keV = th_y.theory_P_lyas(4.0, '21cm_r2_wdm4', 'gadget_ave8_4keV')
r3_ave_4keV = th_y.theory_P_lyas(4.0, '21cm_r3_wdm4', 'gadget_ave8_4keV')
r4_ave_4keV = th_y.theory_P_lyas(4.0, '21cm_r4_wdm4', 'gadget_ave8_4keV')

r1_ave_6keV = th_y.theory_P_lyas(6.0, '21cm_r1_wdm6', 'gadget_ave8_6keV')
r2_ave_6keV = th_y.theory_P_lyas(6.0, '21cm_r2_wdm6', 'gadget_ave8_6keV')
r3_ave_6keV = th_y.theory_P_lyas(6.0, '21cm_r3_wdm6', 'gadget_ave8_6keV')
r4_ave_6keV = th_y.theory_P_lyas(6.0, '21cm_r4_wdm6', 'gadget_ave8_6keV')

r1_ave_9keV = th_y.theory_P_lyas(9.0, '21cm_r1_wdm9', 'gadget_ave8_9keV')
r2_ave_9keV = th_y.theory_P_lyas(9.0, '21cm_r2_wdm9', 'gadget_ave8_9keV')
r3_ave_9keV = th_y.theory_P_lyas(9.0, '21cm_r3_wdm9', 'gadget_ave8_9keV')
r4_ave_9keV = th_y.theory_P_lyas(9.0, '21cm_r4_wdm9', 'gadget_ave8_9keV')

# need to grab average 21fast and all gadget realizations
ave_r1_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r1_cdm')
ave_r2_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r2_cdm')
ave_r3_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r3_cdm')
ave_r4_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r4_cdm')
ave_r5_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r5_cdm')
ave_r6_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r6_cdm')
ave_r7_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r7_cdm')
ave_r8_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_r8_cdm')

ave_r1_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r1_3keV')
ave_r2_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r2_3keV')
ave_r3_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r3_3keV')
ave_r4_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r4_3keV')
ave_r5_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r5_3keV')
ave_r6_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r6_3keV')
ave_r7_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r7_3keV')
ave_r8_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_r8_3keV')

ave_r1_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r1_4keV')
ave_r2_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r2_4keV')
ave_r3_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r3_4keV')
ave_r4_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r4_4keV')
ave_r5_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r5_4keV')
ave_r6_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r6_4keV')
ave_r7_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r7_4keV')
ave_r8_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_r8_4keV')

ave_r1_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r1_6keV')
ave_r2_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r2_6keV')
ave_r3_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r3_6keV')
ave_r4_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r4_6keV')
ave_r5_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r5_6keV')
ave_r6_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r6_6keV')
ave_r7_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r7_6keV')
ave_r8_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_r8_6keV')

ave_r1_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r1_9keV')
ave_r2_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r2_9keV')
ave_r3_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r3_9keV')
ave_r4_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r4_9keV')
ave_r5_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r5_9keV')
ave_r6_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r6_9keV')
ave_r7_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r7_9keV')
ave_r8_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_r8_9keV')

# need to grab the overall mean of both, which gives us our true mean value
ave_ave_cdm = th_y.theory_P_lyas(np.infty, '21cm_avg_cdm', 'gadget_ave8_cdm')
ave_ave_3keV = th_y.theory_P_lyas(3.0, '21cm_avg_wdm3', 'gadget_ave8_3keV')
ave_ave_4keV = th_y.theory_P_lyas(4.0, '21cm_avg_wdm4', 'gadget_ave8_4keV')
ave_ave_6keV = th_y.theory_P_lyas(6.0, '21cm_avg_wdm6', 'gadget_ave8_6keV')
ave_ave_9keV = th_y.theory_P_lyas(9.0, '21cm_avg_wdm9', 'gadget_ave8_9keV')

def sample_variance(ave_ave, r1_ave, r2_ave, r3_ave, r4_ave, ave_r1, ave_r2, ave_r3, ave_r4, ave_r5, ave_r6, ave_r7, ave_r8):
    # computes the sample variance on the mean, i.e. our errorbar, for a given quantity of interest.
    # example input would be power spectra for all those realizations
    # error due to 21cmFAST simulations
    error_fast = (r1_ave - ave_ave)**2 + (r2_ave - ave_ave)**2 + (r3_ave - ave_ave)**2 + (r4_ave - ave_ave)**2
    error_fast = np.sqrt(error_fast / 4.)
    # error due to gadget sims
    error_gadget = (ave_r1 - ave_ave)**2 + (ave_r2 - ave_ave)**2 + (ave_r3 - ave_ave)**2 + (ave_r4 - ave_ave)**2 \
    + (ave_r5 - ave_ave)**2 + (ave_r6 - ave_ave)**2 + (ave_r7 - ave_ave)**2 + (ave_r8 - ave_ave)**2
    error_gadget = np.sqrt(error_gadget / 8.)
    # add in quadrature, i.e. assumme that both simulations are totally uncorrelated (as they should!)
    total_error = np.sqrt(error_gadget**2 + error_fast**2)
    return total_error
def frac_diff(tot, reio):
    # computes the fractional difference of wdm model vs cdm 
    return 100. * reio / tot
# play around at two different redshifts, and for close to line-of-sight.
# I think it is better to plot total flux power spectrum and then fractional difference
# k_array = np.linspace(0.06, 1.4, 500)
k_array = ave_ave_cdm.k_21cm
z1 = 2.0 # redshift of observation
z2 = 2.5
z3 = 3.0
z4 = 3.5
z5 = 4.0
# average 
ave_ave_cdm_plot_z1 = np.zeros(len(k_array))
ave_ave_3keV_plot_z1 = np.zeros(len(k_array))
ave_ave_4keV_plot_z1 = np.zeros(len(k_array))
ave_ave_6keV_plot_z1 = np.zeros(len(k_array))
ave_ave_9keV_plot_z1 = np.zeros(len(k_array))

# dummy for error
ave_ave_cdm_error_z1 = np.zeros(len(k_array))
ave_ave_3keV_error_z1 = np.zeros(len(k_array))
ave_ave_4keV_error_z1 = np.zeros(len(k_array))
ave_ave_6keV_error_z1 = np.zeros(len(k_array))
ave_ave_9keV_error_z1 = np.zeros(len(k_array))

# 21cmfast realizations
r1_ave_cdm_plot_z1 = np.zeros(len(k_array))
r1_ave_3keV_plot_z1 = np.zeros(len(k_array))
r1_ave_4keV_plot_z1 = np.zeros(len(k_array))
r1_ave_6keV_plot_z1 = np.zeros(len(k_array))
r1_ave_9keV_plot_z1 = np.zeros(len(k_array))

r2_ave_cdm_plot_z1 = np.zeros(len(k_array))
r2_ave_3keV_plot_z1 = np.zeros(len(k_array))
r2_ave_4keV_plot_z1 = np.zeros(len(k_array))
r2_ave_6keV_plot_z1 = np.zeros(len(k_array))
r2_ave_9keV_plot_z1 = np.zeros(len(k_array))

r3_ave_cdm_plot_z1 = np.zeros(len(k_array))
r3_ave_3keV_plot_z1 = np.zeros(len(k_array))
r3_ave_4keV_plot_z1 = np.zeros(len(k_array))
r3_ave_6keV_plot_z1 = np.zeros(len(k_array))
r3_ave_9keV_plot_z1 = np.zeros(len(k_array))

r4_ave_cdm_plot_z1 = np.zeros(len(k_array))
r4_ave_3keV_plot_z1 = np.zeros(len(k_array))
r4_ave_4keV_plot_z1 = np.zeros(len(k_array))
r4_ave_6keV_plot_z1 = np.zeros(len(k_array))
r4_ave_9keV_plot_z1 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_plot_z1 = np.zeros(len(k_array))
ave_r1_3keV_plot_z1 = np.zeros(len(k_array))
ave_r1_4keV_plot_z1 = np.zeros(len(k_array))
ave_r1_6keV_plot_z1 = np.zeros(len(k_array))
ave_r1_9keV_plot_z1 = np.zeros(len(k_array))

ave_r2_cdm_plot_z1 = np.zeros(len(k_array))
ave_r2_3keV_plot_z1 = np.zeros(len(k_array))
ave_r2_4keV_plot_z1 = np.zeros(len(k_array))
ave_r2_6keV_plot_z1 = np.zeros(len(k_array))
ave_r2_9keV_plot_z1 = np.zeros(len(k_array))

ave_r3_cdm_plot_z1 = np.zeros(len(k_array))
ave_r3_3keV_plot_z1 = np.zeros(len(k_array))
ave_r3_4keV_plot_z1 = np.zeros(len(k_array))
ave_r3_6keV_plot_z1 = np.zeros(len(k_array))
ave_r3_9keV_plot_z1 = np.zeros(len(k_array))

ave_r4_cdm_plot_z1 = np.zeros(len(k_array))
ave_r4_3keV_plot_z1 = np.zeros(len(k_array))
ave_r4_4keV_plot_z1 = np.zeros(len(k_array))
ave_r4_6keV_plot_z1 = np.zeros(len(k_array))
ave_r4_9keV_plot_z1 = np.zeros(len(k_array))

ave_r5_cdm_plot_z1 = np.zeros(len(k_array))
ave_r5_3keV_plot_z1 = np.zeros(len(k_array))
ave_r5_4keV_plot_z1 = np.zeros(len(k_array))
ave_r5_6keV_plot_z1 = np.zeros(len(k_array))
ave_r5_9keV_plot_z1 = np.zeros(len(k_array))

ave_r6_cdm_plot_z1 = np.zeros(len(k_array))
ave_r6_3keV_plot_z1 = np.zeros(len(k_array))
ave_r6_4keV_plot_z1 = np.zeros(len(k_array))
ave_r6_6keV_plot_z1 = np.zeros(len(k_array))
ave_r6_9keV_plot_z1 = np.zeros(len(k_array))

ave_r7_cdm_plot_z1 = np.zeros(len(k_array))
ave_r7_3keV_plot_z1 = np.zeros(len(k_array))
ave_r7_4keV_plot_z1 = np.zeros(len(k_array))
ave_r7_6keV_plot_z1 = np.zeros(len(k_array))
ave_r7_9keV_plot_z1 = np.zeros(len(k_array))

ave_r8_cdm_plot_z1 = np.zeros(len(k_array))
ave_r8_3keV_plot_z1 = np.zeros(len(k_array))
ave_r8_4keV_plot_z1 = np.zeros(len(k_array))
ave_r8_6keV_plot_z1 = np.zeros(len(k_array))
ave_r8_9keV_plot_z1 = np.zeros(len(k_array))

# z2
ave_ave_cdm_plot_z2 = np.zeros(len(k_array))
ave_ave_3keV_plot_z2 = np.zeros(len(k_array))
ave_ave_4keV_plot_z2 = np.zeros(len(k_array))
ave_ave_6keV_plot_z2 = np.zeros(len(k_array))
ave_ave_9keV_plot_z2 = np.zeros(len(k_array))

# dummy for error
ave_ave_cdm_error_z2 = np.zeros(len(k_array))
ave_ave_3keV_error_z2 = np.zeros(len(k_array))
ave_ave_4keV_error_z2 = np.zeros(len(k_array))
ave_ave_6keV_error_z2 = np.zeros(len(k_array))
ave_ave_9keV_error_z2 = np.zeros(len(k_array))

# 21cmfast realizations
r1_ave_cdm_plot_z2 = np.zeros(len(k_array))
r1_ave_3keV_plot_z2 = np.zeros(len(k_array))
r1_ave_4keV_plot_z2 = np.zeros(len(k_array))
r1_ave_6keV_plot_z2 = np.zeros(len(k_array))
r1_ave_9keV_plot_z2 = np.zeros(len(k_array))

r2_ave_cdm_plot_z2 = np.zeros(len(k_array))
r2_ave_3keV_plot_z2 = np.zeros(len(k_array))
r2_ave_4keV_plot_z2 = np.zeros(len(k_array))
r2_ave_6keV_plot_z2 = np.zeros(len(k_array))
r2_ave_9keV_plot_z2 = np.zeros(len(k_array))

r3_ave_cdm_plot_z2 = np.zeros(len(k_array))
r3_ave_3keV_plot_z2 = np.zeros(len(k_array))
r3_ave_4keV_plot_z2 = np.zeros(len(k_array))
r3_ave_6keV_plot_z2 = np.zeros(len(k_array))
r3_ave_9keV_plot_z2 = np.zeros(len(k_array))

r4_ave_cdm_plot_z2 = np.zeros(len(k_array))
r4_ave_3keV_plot_z2 = np.zeros(len(k_array))
r4_ave_4keV_plot_z2 = np.zeros(len(k_array))
r4_ave_6keV_plot_z2 = np.zeros(len(k_array))
r4_ave_9keV_plot_z2 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_plot_z2 = np.zeros(len(k_array))
ave_r1_3keV_plot_z2 = np.zeros(len(k_array))
ave_r1_4keV_plot_z2 = np.zeros(len(k_array))
ave_r1_6keV_plot_z2 = np.zeros(len(k_array))
ave_r1_9keV_plot_z2 = np.zeros(len(k_array))

ave_r2_cdm_plot_z2 = np.zeros(len(k_array))
ave_r2_3keV_plot_z2 = np.zeros(len(k_array))
ave_r2_4keV_plot_z2 = np.zeros(len(k_array))
ave_r2_6keV_plot_z2 = np.zeros(len(k_array))
ave_r2_9keV_plot_z2 = np.zeros(len(k_array))

ave_r3_cdm_plot_z2 = np.zeros(len(k_array))
ave_r3_3keV_plot_z2 = np.zeros(len(k_array))
ave_r3_4keV_plot_z2 = np.zeros(len(k_array))
ave_r3_6keV_plot_z2 = np.zeros(len(k_array))
ave_r3_9keV_plot_z2 = np.zeros(len(k_array))

ave_r4_cdm_plot_z2 = np.zeros(len(k_array))
ave_r4_3keV_plot_z2 = np.zeros(len(k_array))
ave_r4_4keV_plot_z2 = np.zeros(len(k_array))
ave_r4_6keV_plot_z2 = np.zeros(len(k_array))
ave_r4_9keV_plot_z2 = np.zeros(len(k_array))

ave_r5_cdm_plot_z2 = np.zeros(len(k_array))
ave_r5_3keV_plot_z2 = np.zeros(len(k_array))
ave_r5_4keV_plot_z2 = np.zeros(len(k_array))
ave_r5_6keV_plot_z2 = np.zeros(len(k_array))
ave_r5_9keV_plot_z2 = np.zeros(len(k_array))

ave_r6_cdm_plot_z2 = np.zeros(len(k_array))
ave_r6_3keV_plot_z2 = np.zeros(len(k_array))
ave_r6_4keV_plot_z2 = np.zeros(len(k_array))
ave_r6_6keV_plot_z2 = np.zeros(len(k_array))
ave_r6_9keV_plot_z2 = np.zeros(len(k_array))

ave_r7_cdm_plot_z2 = np.zeros(len(k_array))
ave_r7_3keV_plot_z2 = np.zeros(len(k_array))
ave_r7_4keV_plot_z2 = np.zeros(len(k_array))
ave_r7_6keV_plot_z2 = np.zeros(len(k_array))
ave_r7_9keV_plot_z2 = np.zeros(len(k_array))

ave_r8_cdm_plot_z2 = np.zeros(len(k_array))
ave_r8_3keV_plot_z2 = np.zeros(len(k_array))
ave_r8_4keV_plot_z2 = np.zeros(len(k_array))
ave_r8_6keV_plot_z2 = np.zeros(len(k_array))
ave_r8_9keV_plot_z2 = np.zeros(len(k_array))

# z3

ave_ave_cdm_plot_z3 = np.zeros(len(k_array))
ave_ave_3keV_plot_z3 = np.zeros(len(k_array))
ave_ave_4keV_plot_z3 = np.zeros(len(k_array))
ave_ave_6keV_plot_z3 = np.zeros(len(k_array))
ave_ave_9keV_plot_z3 = np.zeros(len(k_array))

# dummy for error
ave_ave_cdm_error_z3 = np.zeros(len(k_array))
ave_ave_3keV_error_z3 = np.zeros(len(k_array))
ave_ave_4keV_error_z3 = np.zeros(len(k_array))
ave_ave_6keV_error_z3 = np.zeros(len(k_array))
ave_ave_9keV_error_z3 = np.zeros(len(k_array))

# 21cmfast realizations
r1_ave_cdm_plot_z3 = np.zeros(len(k_array))
r1_ave_3keV_plot_z3 = np.zeros(len(k_array))
r1_ave_4keV_plot_z3 = np.zeros(len(k_array))
r1_ave_6keV_plot_z3 = np.zeros(len(k_array))
r1_ave_9keV_plot_z3 = np.zeros(len(k_array))

r2_ave_cdm_plot_z3 = np.zeros(len(k_array))
r2_ave_3keV_plot_z3 = np.zeros(len(k_array))
r2_ave_4keV_plot_z3 = np.zeros(len(k_array))
r2_ave_6keV_plot_z3 = np.zeros(len(k_array))
r2_ave_9keV_plot_z3 = np.zeros(len(k_array))

r3_ave_cdm_plot_z3 = np.zeros(len(k_array))
r3_ave_3keV_plot_z3 = np.zeros(len(k_array))
r3_ave_4keV_plot_z3 = np.zeros(len(k_array))
r3_ave_6keV_plot_z3 = np.zeros(len(k_array))
r3_ave_9keV_plot_z3 = np.zeros(len(k_array))

r4_ave_cdm_plot_z3 = np.zeros(len(k_array))
r4_ave_3keV_plot_z3 = np.zeros(len(k_array))
r4_ave_4keV_plot_z3 = np.zeros(len(k_array))
r4_ave_6keV_plot_z3 = np.zeros(len(k_array))
r4_ave_9keV_plot_z3 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_plot_z3 = np.zeros(len(k_array))
ave_r1_3keV_plot_z3 = np.zeros(len(k_array))
ave_r1_4keV_plot_z3 = np.zeros(len(k_array))
ave_r1_6keV_plot_z3 = np.zeros(len(k_array))
ave_r1_9keV_plot_z3 = np.zeros(len(k_array))

ave_r2_cdm_plot_z3 = np.zeros(len(k_array))
ave_r2_3keV_plot_z3 = np.zeros(len(k_array))
ave_r2_4keV_plot_z3 = np.zeros(len(k_array))
ave_r2_6keV_plot_z3 = np.zeros(len(k_array))
ave_r2_9keV_plot_z3 = np.zeros(len(k_array))

ave_r3_cdm_plot_z3 = np.zeros(len(k_array))
ave_r3_3keV_plot_z3 = np.zeros(len(k_array))
ave_r3_4keV_plot_z3 = np.zeros(len(k_array))
ave_r3_6keV_plot_z3 = np.zeros(len(k_array))
ave_r3_9keV_plot_z3 = np.zeros(len(k_array))

ave_r4_cdm_plot_z3 = np.zeros(len(k_array))
ave_r4_3keV_plot_z3 = np.zeros(len(k_array))
ave_r4_4keV_plot_z3 = np.zeros(len(k_array))
ave_r4_6keV_plot_z3 = np.zeros(len(k_array))
ave_r4_9keV_plot_z3 = np.zeros(len(k_array))

ave_r5_cdm_plot_z3 = np.zeros(len(k_array))
ave_r5_3keV_plot_z3 = np.zeros(len(k_array))
ave_r5_4keV_plot_z3 = np.zeros(len(k_array))
ave_r5_6keV_plot_z3 = np.zeros(len(k_array))
ave_r5_9keV_plot_z3 = np.zeros(len(k_array))

ave_r6_cdm_plot_z3 = np.zeros(len(k_array))
ave_r6_3keV_plot_z3 = np.zeros(len(k_array))
ave_r6_4keV_plot_z3 = np.zeros(len(k_array))
ave_r6_6keV_plot_z3 = np.zeros(len(k_array))
ave_r6_9keV_plot_z3 = np.zeros(len(k_array))

ave_r7_cdm_plot_z3 = np.zeros(len(k_array))
ave_r7_3keV_plot_z3 = np.zeros(len(k_array))
ave_r7_4keV_plot_z3 = np.zeros(len(k_array))
ave_r7_6keV_plot_z3 = np.zeros(len(k_array))
ave_r7_9keV_plot_z3 = np.zeros(len(k_array))

ave_r8_cdm_plot_z3 = np.zeros(len(k_array))
ave_r8_3keV_plot_z3 = np.zeros(len(k_array))
ave_r8_4keV_plot_z3 = np.zeros(len(k_array))
ave_r8_6keV_plot_z3 = np.zeros(len(k_array))
ave_r8_9keV_plot_z3 = np.zeros(len(k_array))

# z4

ave_ave_cdm_plot_z4 = np.zeros(len(k_array))
ave_ave_3keV_plot_z4 = np.zeros(len(k_array))
ave_ave_4keV_plot_z4 = np.zeros(len(k_array))
ave_ave_6keV_plot_z4 = np.zeros(len(k_array))
ave_ave_9keV_plot_z4 = np.zeros(len(k_array))

# dummy for error
ave_ave_cdm_error_z4 = np.zeros(len(k_array))
ave_ave_3keV_error_z4 = np.zeros(len(k_array))
ave_ave_4keV_error_z4 = np.zeros(len(k_array))
ave_ave_6keV_error_z4 = np.zeros(len(k_array))
ave_ave_9keV_error_z4 = np.zeros(len(k_array))

# 21cmfast realizations
r1_ave_cdm_plot_z4 = np.zeros(len(k_array))
r1_ave_3keV_plot_z4 = np.zeros(len(k_array))
r1_ave_4keV_plot_z4 = np.zeros(len(k_array))
r1_ave_6keV_plot_z4 = np.zeros(len(k_array))
r1_ave_9keV_plot_z4 = np.zeros(len(k_array))

r2_ave_cdm_plot_z4 = np.zeros(len(k_array))
r2_ave_3keV_plot_z4 = np.zeros(len(k_array))
r2_ave_4keV_plot_z4 = np.zeros(len(k_array))
r2_ave_6keV_plot_z4 = np.zeros(len(k_array))
r2_ave_9keV_plot_z4 = np.zeros(len(k_array))

r3_ave_cdm_plot_z4 = np.zeros(len(k_array))
r3_ave_3keV_plot_z4 = np.zeros(len(k_array))
r3_ave_4keV_plot_z4 = np.zeros(len(k_array))
r3_ave_6keV_plot_z4 = np.zeros(len(k_array))
r3_ave_9keV_plot_z4 = np.zeros(len(k_array))

r4_ave_cdm_plot_z4 = np.zeros(len(k_array))
r4_ave_3keV_plot_z4 = np.zeros(len(k_array))
r4_ave_4keV_plot_z4 = np.zeros(len(k_array))
r4_ave_6keV_plot_z4 = np.zeros(len(k_array))
r4_ave_9keV_plot_z4 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_plot_z4 = np.zeros(len(k_array))
ave_r1_3keV_plot_z4 = np.zeros(len(k_array))
ave_r1_4keV_plot_z4 = np.zeros(len(k_array))
ave_r1_6keV_plot_z4 = np.zeros(len(k_array))
ave_r1_9keV_plot_z4 = np.zeros(len(k_array))

ave_r2_cdm_plot_z4 = np.zeros(len(k_array))
ave_r2_3keV_plot_z4 = np.zeros(len(k_array))
ave_r2_4keV_plot_z4 = np.zeros(len(k_array))
ave_r2_6keV_plot_z4 = np.zeros(len(k_array))
ave_r2_9keV_plot_z4 = np.zeros(len(k_array))

ave_r3_cdm_plot_z4 = np.zeros(len(k_array))
ave_r3_3keV_plot_z4 = np.zeros(len(k_array))
ave_r3_4keV_plot_z4 = np.zeros(len(k_array))
ave_r3_6keV_plot_z4 = np.zeros(len(k_array))
ave_r3_9keV_plot_z4 = np.zeros(len(k_array))

ave_r4_cdm_plot_z4 = np.zeros(len(k_array))
ave_r4_3keV_plot_z4 = np.zeros(len(k_array))
ave_r4_4keV_plot_z4 = np.zeros(len(k_array))
ave_r4_6keV_plot_z4 = np.zeros(len(k_array))
ave_r4_9keV_plot_z4 = np.zeros(len(k_array))

ave_r5_cdm_plot_z4 = np.zeros(len(k_array))
ave_r5_3keV_plot_z4 = np.zeros(len(k_array))
ave_r5_4keV_plot_z4 = np.zeros(len(k_array))
ave_r5_6keV_plot_z4 = np.zeros(len(k_array))
ave_r5_9keV_plot_z4 = np.zeros(len(k_array))

ave_r6_cdm_plot_z4 = np.zeros(len(k_array))
ave_r6_3keV_plot_z4 = np.zeros(len(k_array))
ave_r6_4keV_plot_z4 = np.zeros(len(k_array))
ave_r6_6keV_plot_z4 = np.zeros(len(k_array))
ave_r6_9keV_plot_z4 = np.zeros(len(k_array))

ave_r7_cdm_plot_z4 = np.zeros(len(k_array))
ave_r7_3keV_plot_z4 = np.zeros(len(k_array))
ave_r7_4keV_plot_z4 = np.zeros(len(k_array))
ave_r7_6keV_plot_z4 = np.zeros(len(k_array))
ave_r7_9keV_plot_z4 = np.zeros(len(k_array))

ave_r8_cdm_plot_z4 = np.zeros(len(k_array))
ave_r8_3keV_plot_z4 = np.zeros(len(k_array))
ave_r8_4keV_plot_z4 = np.zeros(len(k_array))
ave_r8_6keV_plot_z4 = np.zeros(len(k_array))
ave_r8_9keV_plot_z4 = np.zeros(len(k_array))

# z5
ave_ave_cdm_plot_z5 = np.zeros(len(k_array))
ave_ave_3keV_plot_z5 = np.zeros(len(k_array))
ave_ave_4keV_plot_z5 = np.zeros(len(k_array))
ave_ave_6keV_plot_z5 = np.zeros(len(k_array))
ave_ave_9keV_plot_z5 = np.zeros(len(k_array))

# dummy for error
ave_ave_cdm_error_z5 = np.zeros(len(k_array))
ave_ave_3keV_error_z5 = np.zeros(len(k_array))
ave_ave_4keV_error_z5 = np.zeros(len(k_array))
ave_ave_6keV_error_z5 = np.zeros(len(k_array))
ave_ave_9keV_error_z5 = np.zeros(len(k_array))

# 21cmfast realizations
r1_ave_cdm_plot_z5 = np.zeros(len(k_array))
r1_ave_3keV_plot_z5 = np.zeros(len(k_array))
r1_ave_4keV_plot_z5 = np.zeros(len(k_array))
r1_ave_6keV_plot_z5 = np.zeros(len(k_array))
r1_ave_9keV_plot_z5 = np.zeros(len(k_array))

r2_ave_cdm_plot_z5 = np.zeros(len(k_array))
r2_ave_3keV_plot_z5 = np.zeros(len(k_array))
r2_ave_4keV_plot_z5 = np.zeros(len(k_array))
r2_ave_6keV_plot_z5 = np.zeros(len(k_array))
r2_ave_9keV_plot_z5 = np.zeros(len(k_array))

r3_ave_cdm_plot_z5 = np.zeros(len(k_array))
r3_ave_3keV_plot_z5 = np.zeros(len(k_array))
r3_ave_4keV_plot_z5 = np.zeros(len(k_array))
r3_ave_6keV_plot_z5 = np.zeros(len(k_array))
r3_ave_9keV_plot_z5 = np.zeros(len(k_array))

r4_ave_cdm_plot_z5 = np.zeros(len(k_array))
r4_ave_3keV_plot_z5 = np.zeros(len(k_array))
r4_ave_4keV_plot_z5 = np.zeros(len(k_array))
r4_ave_6keV_plot_z5 = np.zeros(len(k_array))
r4_ave_9keV_plot_z5 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_plot_z5 = np.zeros(len(k_array))
ave_r1_3keV_plot_z5 = np.zeros(len(k_array))
ave_r1_4keV_plot_z5 = np.zeros(len(k_array))
ave_r1_6keV_plot_z5 = np.zeros(len(k_array))
ave_r1_9keV_plot_z5 = np.zeros(len(k_array))

ave_r2_cdm_plot_z5 = np.zeros(len(k_array))
ave_r2_3keV_plot_z5 = np.zeros(len(k_array))
ave_r2_4keV_plot_z5 = np.zeros(len(k_array))
ave_r2_6keV_plot_z5 = np.zeros(len(k_array))
ave_r2_9keV_plot_z5 = np.zeros(len(k_array))

ave_r3_cdm_plot_z5 = np.zeros(len(k_array))
ave_r3_3keV_plot_z5 = np.zeros(len(k_array))
ave_r3_4keV_plot_z5 = np.zeros(len(k_array))
ave_r3_6keV_plot_z5 = np.zeros(len(k_array))
ave_r3_9keV_plot_z5 = np.zeros(len(k_array))

ave_r4_cdm_plot_z5 = np.zeros(len(k_array))
ave_r4_3keV_plot_z5 = np.zeros(len(k_array))
ave_r4_4keV_plot_z5 = np.zeros(len(k_array))
ave_r4_6keV_plot_z5 = np.zeros(len(k_array))
ave_r4_9keV_plot_z5 = np.zeros(len(k_array))

ave_r5_cdm_plot_z5 = np.zeros(len(k_array))
ave_r5_3keV_plot_z5 = np.zeros(len(k_array))
ave_r5_4keV_plot_z5 = np.zeros(len(k_array))
ave_r5_6keV_plot_z5 = np.zeros(len(k_array))
ave_r5_9keV_plot_z5 = np.zeros(len(k_array))

ave_r6_cdm_plot_z5 = np.zeros(len(k_array))
ave_r6_3keV_plot_z5 = np.zeros(len(k_array))
ave_r6_4keV_plot_z5 = np.zeros(len(k_array))
ave_r6_6keV_plot_z5 = np.zeros(len(k_array))
ave_r6_9keV_plot_z5 = np.zeros(len(k_array))

ave_r7_cdm_plot_z5 = np.zeros(len(k_array))
ave_r7_3keV_plot_z5 = np.zeros(len(k_array))
ave_r7_4keV_plot_z5 = np.zeros(len(k_array))
ave_r7_6keV_plot_z5 = np.zeros(len(k_array))
ave_r7_9keV_plot_z5 = np.zeros(len(k_array))

ave_r8_cdm_plot_z5 = np.zeros(len(k_array))
ave_r8_3keV_plot_z5 = np.zeros(len(k_array))
ave_r8_4keV_plot_z5 = np.zeros(len(k_array))
ave_r8_6keV_plot_z5 = np.zeros(len(k_array))
ave_r8_9keV_plot_z5 = np.zeros(len(k_array))

# for reionization part only

z1 = 2.0 # redshift of observation
# average 
ave_ave_cdm_reio_z1 = np.zeros(len(k_array))
ave_ave_3keV_reio_z1 = np.zeros(len(k_array))
ave_ave_4keV_reio_z1 = np.zeros(len(k_array))
ave_ave_6keV_reio_z1 = np.zeros(len(k_array))
ave_ave_9keV_reio_z1 = np.zeros(len(k_array))


# 21cmfast realizations
r1_ave_cdm_reio_z1 = np.zeros(len(k_array))
r1_ave_3keV_reio_z1 = np.zeros(len(k_array))
r1_ave_4keV_reio_z1 = np.zeros(len(k_array))
r1_ave_6keV_reio_z1 = np.zeros(len(k_array))
r1_ave_9keV_reio_z1 = np.zeros(len(k_array))

r2_ave_cdm_reio_z1 = np.zeros(len(k_array))
r2_ave_3keV_reio_z1 = np.zeros(len(k_array))
r2_ave_4keV_reio_z1 = np.zeros(len(k_array))
r2_ave_6keV_reio_z1 = np.zeros(len(k_array))
r2_ave_9keV_reio_z1 = np.zeros(len(k_array))

r3_ave_cdm_reio_z1 = np.zeros(len(k_array))
r3_ave_3keV_reio_z1 = np.zeros(len(k_array))
r3_ave_4keV_reio_z1 = np.zeros(len(k_array))
r3_ave_6keV_reio_z1 = np.zeros(len(k_array))
r3_ave_9keV_reio_z1 = np.zeros(len(k_array))

r4_ave_cdm_reio_z1 = np.zeros(len(k_array))
r4_ave_3keV_reio_z1 = np.zeros(len(k_array))
r4_ave_4keV_reio_z1 = np.zeros(len(k_array))
r4_ave_6keV_reio_z1 = np.zeros(len(k_array))
r4_ave_9keV_reio_z1 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_reio_z1 = np.zeros(len(k_array))
ave_r1_3keV_reio_z1 = np.zeros(len(k_array))
ave_r1_4keV_reio_z1 = np.zeros(len(k_array))
ave_r1_6keV_reio_z1 = np.zeros(len(k_array))
ave_r1_9keV_reio_z1 = np.zeros(len(k_array))

ave_r2_cdm_reio_z1 = np.zeros(len(k_array))
ave_r2_3keV_reio_z1 = np.zeros(len(k_array))
ave_r2_4keV_reio_z1 = np.zeros(len(k_array))
ave_r2_6keV_reio_z1 = np.zeros(len(k_array))
ave_r2_9keV_reio_z1 = np.zeros(len(k_array))

ave_r3_cdm_reio_z1 = np.zeros(len(k_array))
ave_r3_3keV_reio_z1 = np.zeros(len(k_array))
ave_r3_4keV_reio_z1 = np.zeros(len(k_array))
ave_r3_6keV_reio_z1 = np.zeros(len(k_array))
ave_r3_9keV_reio_z1 = np.zeros(len(k_array))

ave_r4_cdm_reio_z1 = np.zeros(len(k_array))
ave_r4_3keV_reio_z1 = np.zeros(len(k_array))
ave_r4_4keV_reio_z1 = np.zeros(len(k_array))
ave_r4_6keV_reio_z1 = np.zeros(len(k_array))
ave_r4_9keV_reio_z1 = np.zeros(len(k_array))

ave_r5_cdm_reio_z1 = np.zeros(len(k_array))
ave_r5_3keV_reio_z1 = np.zeros(len(k_array))
ave_r5_4keV_reio_z1 = np.zeros(len(k_array))
ave_r5_6keV_reio_z1 = np.zeros(len(k_array))
ave_r5_9keV_reio_z1 = np.zeros(len(k_array))

ave_r6_cdm_reio_z1 = np.zeros(len(k_array))
ave_r6_3keV_reio_z1 = np.zeros(len(k_array))
ave_r6_4keV_reio_z1 = np.zeros(len(k_array))
ave_r6_6keV_reio_z1 = np.zeros(len(k_array))
ave_r6_9keV_reio_z1 = np.zeros(len(k_array))

ave_r7_cdm_reio_z1 = np.zeros(len(k_array))
ave_r7_3keV_reio_z1 = np.zeros(len(k_array))
ave_r7_4keV_reio_z1 = np.zeros(len(k_array))
ave_r7_6keV_reio_z1 = np.zeros(len(k_array))
ave_r7_9keV_reio_z1 = np.zeros(len(k_array))

ave_r8_cdm_reio_z1 = np.zeros(len(k_array))
ave_r8_3keV_reio_z1 = np.zeros(len(k_array))
ave_r8_4keV_reio_z1 = np.zeros(len(k_array))
ave_r8_6keV_reio_z1 = np.zeros(len(k_array))
ave_r8_9keV_reio_z1 = np.zeros(len(k_array))

# z2
ave_ave_cdm_reio_z2 = np.zeros(len(k_array))
ave_ave_3keV_reio_z2 = np.zeros(len(k_array))
ave_ave_4keV_reio_z2 = np.zeros(len(k_array))
ave_ave_6keV_reio_z2 = np.zeros(len(k_array))
ave_ave_9keV_reio_z2 = np.zeros(len(k_array))


# 21cmfast realizations
r1_ave_cdm_reio_z2 = np.zeros(len(k_array))
r1_ave_3keV_reio_z2 = np.zeros(len(k_array))
r1_ave_4keV_reio_z2 = np.zeros(len(k_array))
r1_ave_6keV_reio_z2 = np.zeros(len(k_array))
r1_ave_9keV_reio_z2 = np.zeros(len(k_array))

r2_ave_cdm_reio_z2 = np.zeros(len(k_array))
r2_ave_3keV_reio_z2 = np.zeros(len(k_array))
r2_ave_4keV_reio_z2 = np.zeros(len(k_array))
r2_ave_6keV_reio_z2 = np.zeros(len(k_array))
r2_ave_9keV_reio_z2 = np.zeros(len(k_array))

r3_ave_cdm_reio_z2 = np.zeros(len(k_array))
r3_ave_3keV_reio_z2 = np.zeros(len(k_array))
r3_ave_4keV_reio_z2 = np.zeros(len(k_array))
r3_ave_6keV_reio_z2 = np.zeros(len(k_array))
r3_ave_9keV_reio_z2 = np.zeros(len(k_array))

r4_ave_cdm_reio_z2 = np.zeros(len(k_array))
r4_ave_3keV_reio_z2 = np.zeros(len(k_array))
r4_ave_4keV_reio_z2 = np.zeros(len(k_array))
r4_ave_6keV_reio_z2 = np.zeros(len(k_array))
r4_ave_9keV_reio_z2 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_reio_z2 = np.zeros(len(k_array))
ave_r1_3keV_reio_z2 = np.zeros(len(k_array))
ave_r1_4keV_reio_z2 = np.zeros(len(k_array))
ave_r1_6keV_reio_z2 = np.zeros(len(k_array))
ave_r1_9keV_reio_z2 = np.zeros(len(k_array))

ave_r2_cdm_reio_z2 = np.zeros(len(k_array))
ave_r2_3keV_reio_z2 = np.zeros(len(k_array))
ave_r2_4keV_reio_z2 = np.zeros(len(k_array))
ave_r2_6keV_reio_z2 = np.zeros(len(k_array))
ave_r2_9keV_reio_z2 = np.zeros(len(k_array))

ave_r3_cdm_reio_z2 = np.zeros(len(k_array))
ave_r3_3keV_reio_z2 = np.zeros(len(k_array))
ave_r3_4keV_reio_z2 = np.zeros(len(k_array))
ave_r3_6keV_reio_z2 = np.zeros(len(k_array))
ave_r3_9keV_reio_z2 = np.zeros(len(k_array))

ave_r4_cdm_reio_z2 = np.zeros(len(k_array))
ave_r4_3keV_reio_z2 = np.zeros(len(k_array))
ave_r4_4keV_reio_z2 = np.zeros(len(k_array))
ave_r4_6keV_reio_z2 = np.zeros(len(k_array))
ave_r4_9keV_reio_z2 = np.zeros(len(k_array))

ave_r5_cdm_reio_z2 = np.zeros(len(k_array))
ave_r5_3keV_reio_z2 = np.zeros(len(k_array))
ave_r5_4keV_reio_z2 = np.zeros(len(k_array))
ave_r5_6keV_reio_z2 = np.zeros(len(k_array))
ave_r5_9keV_reio_z2 = np.zeros(len(k_array))

ave_r6_cdm_reio_z2 = np.zeros(len(k_array))
ave_r6_3keV_reio_z2 = np.zeros(len(k_array))
ave_r6_4keV_reio_z2 = np.zeros(len(k_array))
ave_r6_6keV_reio_z2 = np.zeros(len(k_array))
ave_r6_9keV_reio_z2 = np.zeros(len(k_array))

ave_r7_cdm_reio_z2 = np.zeros(len(k_array))
ave_r7_3keV_reio_z2 = np.zeros(len(k_array))
ave_r7_4keV_reio_z2 = np.zeros(len(k_array))
ave_r7_6keV_reio_z2 = np.zeros(len(k_array))
ave_r7_9keV_reio_z2 = np.zeros(len(k_array))

ave_r8_cdm_reio_z2 = np.zeros(len(k_array))
ave_r8_3keV_reio_z2 = np.zeros(len(k_array))
ave_r8_4keV_reio_z2 = np.zeros(len(k_array))
ave_r8_6keV_reio_z2 = np.zeros(len(k_array))
ave_r8_9keV_reio_z2 = np.zeros(len(k_array))

# z3

ave_ave_cdm_reio_z3 = np.zeros(len(k_array))
ave_ave_3keV_reio_z3 = np.zeros(len(k_array))
ave_ave_4keV_reio_z3 = np.zeros(len(k_array))
ave_ave_6keV_reio_z3 = np.zeros(len(k_array))
ave_ave_9keV_reio_z3 = np.zeros(len(k_array))


# 21cmfast realizations
r1_ave_cdm_reio_z3 = np.zeros(len(k_array))
r1_ave_3keV_reio_z3 = np.zeros(len(k_array))
r1_ave_4keV_reio_z3 = np.zeros(len(k_array))
r1_ave_6keV_reio_z3 = np.zeros(len(k_array))
r1_ave_9keV_reio_z3 = np.zeros(len(k_array))

r2_ave_cdm_reio_z3 = np.zeros(len(k_array))
r2_ave_3keV_reio_z3 = np.zeros(len(k_array))
r2_ave_4keV_reio_z3 = np.zeros(len(k_array))
r2_ave_6keV_reio_z3 = np.zeros(len(k_array))
r2_ave_9keV_reio_z3 = np.zeros(len(k_array))

r3_ave_cdm_reio_z3 = np.zeros(len(k_array))
r3_ave_3keV_reio_z3 = np.zeros(len(k_array))
r3_ave_4keV_reio_z3 = np.zeros(len(k_array))
r3_ave_6keV_reio_z3 = np.zeros(len(k_array))
r3_ave_9keV_reio_z3 = np.zeros(len(k_array))

r4_ave_cdm_reio_z3 = np.zeros(len(k_array))
r4_ave_3keV_reio_z3 = np.zeros(len(k_array))
r4_ave_4keV_reio_z3 = np.zeros(len(k_array))
r4_ave_6keV_reio_z3 = np.zeros(len(k_array))
r4_ave_9keV_reio_z3 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_reio_z3 = np.zeros(len(k_array))
ave_r1_3keV_reio_z3 = np.zeros(len(k_array))
ave_r1_4keV_reio_z3 = np.zeros(len(k_array))
ave_r1_6keV_reio_z3 = np.zeros(len(k_array))
ave_r1_9keV_reio_z3 = np.zeros(len(k_array))

ave_r2_cdm_reio_z3 = np.zeros(len(k_array))
ave_r2_3keV_reio_z3 = np.zeros(len(k_array))
ave_r2_4keV_reio_z3 = np.zeros(len(k_array))
ave_r2_6keV_reio_z3 = np.zeros(len(k_array))
ave_r2_9keV_reio_z3 = np.zeros(len(k_array))

ave_r3_cdm_reio_z3 = np.zeros(len(k_array))
ave_r3_3keV_reio_z3 = np.zeros(len(k_array))
ave_r3_4keV_reio_z3 = np.zeros(len(k_array))
ave_r3_6keV_reio_z3 = np.zeros(len(k_array))
ave_r3_9keV_reio_z3 = np.zeros(len(k_array))

ave_r4_cdm_reio_z3 = np.zeros(len(k_array))
ave_r4_3keV_reio_z3 = np.zeros(len(k_array))
ave_r4_4keV_reio_z3 = np.zeros(len(k_array))
ave_r4_6keV_reio_z3 = np.zeros(len(k_array))
ave_r4_9keV_reio_z3 = np.zeros(len(k_array))

ave_r5_cdm_reio_z3 = np.zeros(len(k_array))
ave_r5_3keV_reio_z3 = np.zeros(len(k_array))
ave_r5_4keV_reio_z3 = np.zeros(len(k_array))
ave_r5_6keV_reio_z3 = np.zeros(len(k_array))
ave_r5_9keV_reio_z3 = np.zeros(len(k_array))

ave_r6_cdm_reio_z3 = np.zeros(len(k_array))
ave_r6_3keV_reio_z3 = np.zeros(len(k_array))
ave_r6_4keV_reio_z3 = np.zeros(len(k_array))
ave_r6_6keV_reio_z3 = np.zeros(len(k_array))
ave_r6_9keV_reio_z3 = np.zeros(len(k_array))

ave_r7_cdm_reio_z3 = np.zeros(len(k_array))
ave_r7_3keV_reio_z3 = np.zeros(len(k_array))
ave_r7_4keV_reio_z3 = np.zeros(len(k_array))
ave_r7_6keV_reio_z3 = np.zeros(len(k_array))
ave_r7_9keV_reio_z3 = np.zeros(len(k_array))

ave_r8_cdm_reio_z3 = np.zeros(len(k_array))
ave_r8_3keV_reio_z3 = np.zeros(len(k_array))
ave_r8_4keV_reio_z3 = np.zeros(len(k_array))
ave_r8_6keV_reio_z3 = np.zeros(len(k_array))
ave_r8_9keV_reio_z3 = np.zeros(len(k_array))

# z4

ave_ave_cdm_reio_z4 = np.zeros(len(k_array))
ave_ave_3keV_reio_z4 = np.zeros(len(k_array))
ave_ave_4keV_reio_z4 = np.zeros(len(k_array))
ave_ave_6keV_reio_z4 = np.zeros(len(k_array))
ave_ave_9keV_reio_z4 = np.zeros(len(k_array))


# 21cmfast realizations
r1_ave_cdm_reio_z4 = np.zeros(len(k_array))
r1_ave_3keV_reio_z4 = np.zeros(len(k_array))
r1_ave_4keV_reio_z4 = np.zeros(len(k_array))
r1_ave_6keV_reio_z4 = np.zeros(len(k_array))
r1_ave_9keV_reio_z4 = np.zeros(len(k_array))

r2_ave_cdm_reio_z4 = np.zeros(len(k_array))
r2_ave_3keV_reio_z4 = np.zeros(len(k_array))
r2_ave_4keV_reio_z4 = np.zeros(len(k_array))
r2_ave_6keV_reio_z4 = np.zeros(len(k_array))
r2_ave_9keV_reio_z4 = np.zeros(len(k_array))

r3_ave_cdm_reio_z4 = np.zeros(len(k_array))
r3_ave_3keV_reio_z4 = np.zeros(len(k_array))
r3_ave_4keV_reio_z4 = np.zeros(len(k_array))
r3_ave_6keV_reio_z4 = np.zeros(len(k_array))
r3_ave_9keV_reio_z4 = np.zeros(len(k_array))

r4_ave_cdm_reio_z4 = np.zeros(len(k_array))
r4_ave_3keV_reio_z4 = np.zeros(len(k_array))
r4_ave_4keV_reio_z4 = np.zeros(len(k_array))
r4_ave_6keV_reio_z4 = np.zeros(len(k_array))
r4_ave_9keV_reio_z4 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_reio_z4 = np.zeros(len(k_array))
ave_r1_3keV_reio_z4 = np.zeros(len(k_array))
ave_r1_4keV_reio_z4 = np.zeros(len(k_array))
ave_r1_6keV_reio_z4 = np.zeros(len(k_array))
ave_r1_9keV_reio_z4 = np.zeros(len(k_array))

ave_r2_cdm_reio_z4 = np.zeros(len(k_array))
ave_r2_3keV_reio_z4 = np.zeros(len(k_array))
ave_r2_4keV_reio_z4 = np.zeros(len(k_array))
ave_r2_6keV_reio_z4 = np.zeros(len(k_array))
ave_r2_9keV_reio_z4 = np.zeros(len(k_array))

ave_r3_cdm_reio_z4 = np.zeros(len(k_array))
ave_r3_3keV_reio_z4 = np.zeros(len(k_array))
ave_r3_4keV_reio_z4 = np.zeros(len(k_array))
ave_r3_6keV_reio_z4 = np.zeros(len(k_array))
ave_r3_9keV_reio_z4 = np.zeros(len(k_array))

ave_r4_cdm_reio_z4 = np.zeros(len(k_array))
ave_r4_3keV_reio_z4 = np.zeros(len(k_array))
ave_r4_4keV_reio_z4 = np.zeros(len(k_array))
ave_r4_6keV_reio_z4 = np.zeros(len(k_array))
ave_r4_9keV_reio_z4 = np.zeros(len(k_array))

ave_r5_cdm_reio_z4 = np.zeros(len(k_array))
ave_r5_3keV_reio_z4 = np.zeros(len(k_array))
ave_r5_4keV_reio_z4 = np.zeros(len(k_array))
ave_r5_6keV_reio_z4 = np.zeros(len(k_array))
ave_r5_9keV_reio_z4 = np.zeros(len(k_array))

ave_r6_cdm_reio_z4 = np.zeros(len(k_array))
ave_r6_3keV_reio_z4 = np.zeros(len(k_array))
ave_r6_4keV_reio_z4 = np.zeros(len(k_array))
ave_r6_6keV_reio_z4 = np.zeros(len(k_array))
ave_r6_9keV_reio_z4 = np.zeros(len(k_array))

ave_r7_cdm_reio_z4 = np.zeros(len(k_array))
ave_r7_3keV_reio_z4 = np.zeros(len(k_array))
ave_r7_4keV_reio_z4 = np.zeros(len(k_array))
ave_r7_6keV_reio_z4 = np.zeros(len(k_array))
ave_r7_9keV_reio_z4 = np.zeros(len(k_array))

ave_r8_cdm_reio_z4 = np.zeros(len(k_array))
ave_r8_3keV_reio_z4 = np.zeros(len(k_array))
ave_r8_4keV_reio_z4 = np.zeros(len(k_array))
ave_r8_6keV_reio_z4 = np.zeros(len(k_array))
ave_r8_9keV_reio_z4 = np.zeros(len(k_array))

# next redshift
z5 = 4.0 # redshift of observation
ave_ave_cdm_reio_z5 = np.zeros(len(k_array))
ave_ave_3keV_reio_z5 = np.zeros(len(k_array))
ave_ave_4keV_reio_z5 = np.zeros(len(k_array))
ave_ave_6keV_reio_z5 = np.zeros(len(k_array))
ave_ave_9keV_reio_z5 = np.zeros(len(k_array))

# 21cmfast realizations
r1_ave_cdm_reio_z5 = np.zeros(len(k_array))
r1_ave_3keV_reio_z5 = np.zeros(len(k_array))
r1_ave_4keV_reio_z5 = np.zeros(len(k_array))
r1_ave_6keV_reio_z5 = np.zeros(len(k_array))
r1_ave_9keV_reio_z5 = np.zeros(len(k_array))

r2_ave_cdm_reio_z5 = np.zeros(len(k_array))
r2_ave_3keV_reio_z5 = np.zeros(len(k_array))
r2_ave_4keV_reio_z5 = np.zeros(len(k_array))
r2_ave_6keV_reio_z5 = np.zeros(len(k_array))
r2_ave_9keV_reio_z5 = np.zeros(len(k_array))

r3_ave_cdm_reio_z5 = np.zeros(len(k_array))
r3_ave_3keV_reio_z5 = np.zeros(len(k_array))
r3_ave_4keV_reio_z5 = np.zeros(len(k_array))
r3_ave_6keV_reio_z5 = np.zeros(len(k_array))
r3_ave_9keV_reio_z5 = np.zeros(len(k_array))

r4_ave_cdm_reio_z5 = np.zeros(len(k_array))
r4_ave_3keV_reio_z5 = np.zeros(len(k_array))
r4_ave_4keV_reio_z5 = np.zeros(len(k_array))
r4_ave_6keV_reio_z5 = np.zeros(len(k_array))
r4_ave_9keV_reio_z5 = np.zeros(len(k_array))

# gadget realizations
ave_r1_cdm_reio_z5 = np.zeros(len(k_array))
ave_r1_3keV_reio_z5 = np.zeros(len(k_array))
ave_r1_4keV_reio_z5 = np.zeros(len(k_array))
ave_r1_6keV_reio_z5 = np.zeros(len(k_array))
ave_r1_9keV_reio_z5 = np.zeros(len(k_array))

ave_r2_cdm_reio_z5 = np.zeros(len(k_array))
ave_r2_3keV_reio_z5 = np.zeros(len(k_array))
ave_r2_4keV_reio_z5 = np.zeros(len(k_array))
ave_r2_6keV_reio_z5 = np.zeros(len(k_array))
ave_r2_9keV_reio_z5 = np.zeros(len(k_array))

ave_r3_cdm_reio_z5 = np.zeros(len(k_array))
ave_r3_3keV_reio_z5 = np.zeros(len(k_array))
ave_r3_4keV_reio_z5 = np.zeros(len(k_array))
ave_r3_6keV_reio_z5 = np.zeros(len(k_array))
ave_r3_9keV_reio_z5 = np.zeros(len(k_array))

ave_r4_cdm_reio_z5 = np.zeros(len(k_array))
ave_r4_3keV_reio_z5 = np.zeros(len(k_array))
ave_r4_4keV_reio_z5 = np.zeros(len(k_array))
ave_r4_6keV_reio_z5 = np.zeros(len(k_array))
ave_r4_9keV_reio_z5 = np.zeros(len(k_array))

ave_r5_cdm_reio_z5 = np.zeros(len(k_array))
ave_r5_3keV_reio_z5 = np.zeros(len(k_array))
ave_r5_4keV_reio_z5 = np.zeros(len(k_array))
ave_r5_6keV_reio_z5 = np.zeros(len(k_array))
ave_r5_9keV_reio_z5 = np.zeros(len(k_array))

ave_r6_cdm_reio_z5 = np.zeros(len(k_array))
ave_r6_3keV_reio_z5 = np.zeros(len(k_array))
ave_r6_4keV_reio_z5 = np.zeros(len(k_array))
ave_r6_6keV_reio_z5 = np.zeros(len(k_array))
ave_r6_9keV_reio_z5 = np.zeros(len(k_array))

ave_r7_cdm_reio_z5 = np.zeros(len(k_array))
ave_r7_3keV_reio_z5 = np.zeros(len(k_array))
ave_r7_4keV_reio_z5 = np.zeros(len(k_array))
ave_r7_6keV_reio_z5 = np.zeros(len(k_array))
ave_r7_9keV_reio_z5 = np.zeros(len(k_array))

ave_r8_cdm_reio_z5 = np.zeros(len(k_array))
ave_r8_3keV_reio_z5 = np.zeros(len(k_array))
ave_r8_4keV_reio_z5 = np.zeros(len(k_array))
ave_r8_6keV_reio_z5 = np.zeros(len(k_array))
ave_r8_9keV_reio_z5 = np.zeros(len(k_array))

for i in range(0,len(k_array)):
    # frist redshift
    # average
    ave_ave_cdm_plot_z1[i] = ave_ave_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_3keV_plot_z1[i] = ave_ave_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_4keV_plot_z1[i] = ave_ave_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_6keV_plot_z1[i] = ave_ave_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_9keV_plot_z1[i] = ave_ave_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_plot_z1[i] = r1_ave_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r1_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_3keV_plot_z1[i] = r1_ave_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r1_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_4keV_plot_z1[i] = r1_ave_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r1_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_6keV_plot_z1[i] = r1_ave_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r1_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_9keV_plot_z1[i] = r1_ave_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r1_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    
    r2_ave_cdm_plot_z1[i] = r2_ave_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r2_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_3keV_plot_z1[i] = r2_ave_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r2_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_4keV_plot_z1[i] = r2_ave_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r2_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_6keV_plot_z1[i] = r2_ave_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r2_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_9keV_plot_z1[i] = r2_ave_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r2_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    
    r3_ave_cdm_plot_z1[i] = r3_ave_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r3_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_3keV_plot_z1[i] = r3_ave_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r3_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_4keV_plot_z1[i] = r3_ave_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r3_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_6keV_plot_z1[i] = r3_ave_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r3_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_9keV_plot_z1[i] = r3_ave_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r3_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    r4_ave_cdm_plot_z1[i] = r4_ave_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r4_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_3keV_plot_z1[i] = r4_ave_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r4_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_4keV_plot_z1[i] = r4_ave_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r4_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_6keV_plot_z1[i] = r4_ave_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r4_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_9keV_plot_z1[i] = r4_ave_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + r4_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_plot_z1[i] = ave_r1_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r1_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_3keV_plot_z1[i] = ave_r1_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r1_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_4keV_plot_z1[i] = ave_r1_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r1_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_6keV_plot_z1[i] = ave_r1_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r1_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_9keV_plot_z1[i] = ave_r1_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r1_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r2_cdm_plot_z1[i] = ave_r2_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r2_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_3keV_plot_z1[i] = ave_r2_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r2_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_4keV_plot_z1[i] = ave_r2_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r2_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_6keV_plot_z1[i] = ave_r2_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r2_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_9keV_plot_z1[i] = ave_r2_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r2_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r3_cdm_plot_z1[i] = ave_r3_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r3_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_3keV_plot_z1[i] = ave_r3_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r3_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_4keV_plot_z1[i] = ave_r3_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r3_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_6keV_plot_z1[i] = ave_r3_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r3_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_9keV_plot_z1[i] = ave_r3_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r3_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r4_cdm_plot_z1[i] = ave_r4_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r4_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_3keV_plot_z1[i] = ave_r4_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r4_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_4keV_plot_z1[i] = ave_r4_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r4_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_6keV_plot_z1[i] = ave_r4_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r4_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_9keV_plot_z1[i] = ave_r4_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r4_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r5_cdm_plot_z1[i] = ave_r5_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r5_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_3keV_plot_z1[i] = ave_r5_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r5_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_4keV_plot_z1[i] = ave_r5_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r5_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_6keV_plot_z1[i] = ave_r5_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r5_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_9keV_plot_z1[i] = ave_r5_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r5_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r6_cdm_plot_z1[i] = ave_r6_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r6_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_3keV_plot_z1[i] = ave_r6_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r6_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_4keV_plot_z1[i] = ave_r6_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r6_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_6keV_plot_z1[i] = ave_r6_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r6_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_9keV_plot_z1[i] = ave_r6_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r6_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r7_cdm_plot_z1[i] = ave_r7_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r7_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_3keV_plot_z1[i] = ave_r7_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r7_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_4keV_plot_z1[i] = ave_r7_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r7_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_6keV_plot_z1[i] = ave_r7_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r7_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_9keV_plot_z1[i] = ave_r7_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r7_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r8_cdm_plot_z1[i] = ave_r8_cdm.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r8_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_3keV_plot_z1[i] = ave_r8_3keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r8_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_4keV_plot_z1[i] = ave_r8_4keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r8_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_6keV_plot_z1[i] = ave_r8_6keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r8_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_9keV_plot_z1[i] = ave_r8_9keV.FluxP3D_lya_Mpc(z1, k_array[i], 0.1) + ave_r8_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    # z2
    # average
    ave_ave_cdm_plot_z2[i] = ave_ave_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_3keV_plot_z2[i] = ave_ave_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_4keV_plot_z2[i] = ave_ave_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_6keV_plot_z2[i] = ave_ave_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_9keV_plot_z2[i] = ave_ave_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    # 21cmfast realizations

    r1_ave_cdm_plot_z2[i] = r1_ave_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r1_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_3keV_plot_z2[i] = r1_ave_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r1_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_4keV_plot_z2[i] = r1_ave_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r1_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_6keV_plot_z2[i] = r1_ave_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r1_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_9keV_plot_z2[i] = r1_ave_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r1_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    
    r2_ave_cdm_plot_z2[i] = r2_ave_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r2_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_3keV_plot_z2[i] = r2_ave_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r2_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_4keV_plot_z2[i] = r2_ave_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r2_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_6keV_plot_z2[i] = r2_ave_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r2_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_9keV_plot_z2[i] = r2_ave_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r2_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    
    r3_ave_cdm_plot_z2[i] = r3_ave_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r3_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_3keV_plot_z2[i] = r3_ave_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r3_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_4keV_plot_z2[i] = r3_ave_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r3_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_6keV_plot_z2[i] = r3_ave_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r3_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_9keV_plot_z2[i] = r3_ave_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r3_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    r4_ave_cdm_plot_z2[i] = r4_ave_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r4_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_3keV_plot_z2[i] = r4_ave_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r4_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_4keV_plot_z2[i] = r4_ave_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r4_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_6keV_plot_z2[i] = r4_ave_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r4_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_9keV_plot_z2[i] = r4_ave_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + r4_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_plot_z2[i] = ave_r1_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r1_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_3keV_plot_z2[i] = ave_r1_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r1_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_4keV_plot_z2[i] = ave_r1_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r1_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_6keV_plot_z2[i] = ave_r1_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r1_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_9keV_plot_z2[i] = ave_r1_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r1_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r2_cdm_plot_z2[i] = ave_r2_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r2_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_3keV_plot_z2[i] = ave_r2_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r2_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_4keV_plot_z2[i] = ave_r2_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r2_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_6keV_plot_z2[i] = ave_r2_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r2_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_9keV_plot_z2[i] = ave_r2_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r2_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r3_cdm_plot_z2[i] = ave_r3_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r3_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_3keV_plot_z2[i] = ave_r3_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r3_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_4keV_plot_z2[i] = ave_r3_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r3_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_6keV_plot_z2[i] = ave_r3_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r3_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_9keV_plot_z2[i] = ave_r3_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r3_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r4_cdm_plot_z2[i] = ave_r4_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r4_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_3keV_plot_z2[i] = ave_r4_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r4_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_4keV_plot_z2[i] = ave_r4_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r4_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_6keV_plot_z2[i] = ave_r4_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r4_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_9keV_plot_z2[i] = ave_r4_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r4_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r5_cdm_plot_z2[i] = ave_r5_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r5_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_3keV_plot_z2[i] = ave_r5_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r5_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_4keV_plot_z2[i] = ave_r5_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r5_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_6keV_plot_z2[i] = ave_r5_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r5_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_9keV_plot_z2[i] = ave_r5_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r5_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r6_cdm_plot_z2[i] = ave_r6_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r6_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_3keV_plot_z2[i] = ave_r6_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r6_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_4keV_plot_z2[i] = ave_r6_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r6_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_6keV_plot_z2[i] = ave_r6_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r6_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_9keV_plot_z2[i] = ave_r6_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r6_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r7_cdm_plot_z2[i] = ave_r7_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r7_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_3keV_plot_z2[i] = ave_r7_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r7_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_4keV_plot_z2[i] = ave_r7_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r7_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_6keV_plot_z2[i] = ave_r7_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r7_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_9keV_plot_z2[i] = ave_r7_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r7_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r8_cdm_plot_z2[i] = ave_r8_cdm.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r8_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_3keV_plot_z2[i] = ave_r8_3keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r8_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_4keV_plot_z2[i] = ave_r8_4keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r8_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_6keV_plot_z2[i] = ave_r8_6keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r8_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_9keV_plot_z2[i] = ave_r8_9keV.FluxP3D_lya_Mpc(z2, k_array[i], 0.1) + ave_r8_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    # z3
    # average
    ave_ave_cdm_plot_z3[i] = ave_ave_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_3keV_plot_z3[i] = ave_ave_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_4keV_plot_z3[i] = ave_ave_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_6keV_plot_z3[i] = ave_ave_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_9keV_plot_z3[i] = ave_ave_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    # 21cmfast realizations

    r1_ave_cdm_plot_z3[i] = r1_ave_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r1_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_3keV_plot_z3[i] = r1_ave_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r1_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_4keV_plot_z3[i] = r1_ave_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r1_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_6keV_plot_z3[i] = r1_ave_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r1_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_9keV_plot_z3[i] = r1_ave_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r1_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    
    r2_ave_cdm_plot_z3[i] = r2_ave_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r2_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_3keV_plot_z3[i] = r2_ave_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r2_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_4keV_plot_z3[i] = r2_ave_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r2_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_6keV_plot_z3[i] = r2_ave_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r2_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_9keV_plot_z3[i] = r2_ave_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r2_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    
    r3_ave_cdm_plot_z3[i] = r3_ave_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r3_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_3keV_plot_z3[i] = r3_ave_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r3_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_4keV_plot_z3[i] = r3_ave_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r3_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_6keV_plot_z3[i] = r3_ave_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r3_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_9keV_plot_z3[i] = r3_ave_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r3_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    r4_ave_cdm_plot_z3[i] = r4_ave_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r4_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_3keV_plot_z3[i] = r4_ave_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r4_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_4keV_plot_z3[i] = r4_ave_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r4_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_6keV_plot_z3[i] = r4_ave_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r4_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_9keV_plot_z3[i] = r4_ave_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + r4_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_plot_z3[i] = ave_r1_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r1_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_3keV_plot_z3[i] = ave_r1_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r1_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_4keV_plot_z3[i] = ave_r1_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r1_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_6keV_plot_z3[i] = ave_r1_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r1_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_9keV_plot_z3[i] = ave_r1_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r1_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r2_cdm_plot_z3[i] = ave_r2_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r2_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_3keV_plot_z3[i] = ave_r2_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r2_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_4keV_plot_z3[i] = ave_r2_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r2_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_6keV_plot_z3[i] = ave_r2_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r2_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_9keV_plot_z3[i] = ave_r2_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r2_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r3_cdm_plot_z3[i] = ave_r3_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r3_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_3keV_plot_z3[i] = ave_r3_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r3_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_4keV_plot_z3[i] = ave_r3_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r3_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_6keV_plot_z3[i] = ave_r3_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r3_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_9keV_plot_z3[i] = ave_r3_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r3_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r4_cdm_plot_z3[i] = ave_r4_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r4_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_3keV_plot_z3[i] = ave_r4_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r4_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_4keV_plot_z3[i] = ave_r4_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r4_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_6keV_plot_z3[i] = ave_r4_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r4_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_9keV_plot_z3[i] = ave_r4_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r4_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r5_cdm_plot_z3[i] = ave_r5_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r5_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_3keV_plot_z3[i] = ave_r5_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r5_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_4keV_plot_z3[i] = ave_r5_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r5_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_6keV_plot_z3[i] = ave_r5_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r5_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_9keV_plot_z3[i] = ave_r5_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r5_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r6_cdm_plot_z3[i] = ave_r6_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r6_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_3keV_plot_z3[i] = ave_r6_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r6_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_4keV_plot_z3[i] = ave_r6_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r6_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_6keV_plot_z3[i] = ave_r6_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r6_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_9keV_plot_z3[i] = ave_r6_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r6_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r7_cdm_plot_z3[i] = ave_r7_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r7_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_3keV_plot_z3[i] = ave_r7_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r7_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_4keV_plot_z3[i] = ave_r7_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r7_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_6keV_plot_z3[i] = ave_r7_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r7_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_9keV_plot_z3[i] = ave_r7_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r7_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r8_cdm_plot_z3[i] = ave_r8_cdm.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r8_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_3keV_plot_z3[i] = ave_r8_3keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r8_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_4keV_plot_z3[i] = ave_r8_4keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r8_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_6keV_plot_z3[i] = ave_r8_6keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r8_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_9keV_plot_z3[i] = ave_r8_9keV.FluxP3D_lya_Mpc(z3, k_array[i], 0.1) + ave_r8_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    # z4
    # average
    ave_ave_cdm_plot_z4[i] = ave_ave_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_3keV_plot_z4[i] = ave_ave_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_4keV_plot_z4[i] = ave_ave_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_6keV_plot_z4[i] = ave_ave_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_9keV_plot_z4[i] = ave_ave_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    # 21cmfast realizations

    r1_ave_cdm_plot_z4[i] = r1_ave_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r1_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_3keV_plot_z4[i] = r1_ave_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r1_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_4keV_plot_z4[i] = r1_ave_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r1_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_6keV_plot_z4[i] = r1_ave_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r1_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_9keV_plot_z4[i] = r1_ave_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r1_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    
    r2_ave_cdm_plot_z4[i] = r2_ave_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r2_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_3keV_plot_z4[i] = r2_ave_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r2_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_4keV_plot_z4[i] = r2_ave_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r2_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_6keV_plot_z4[i] = r2_ave_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r2_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_9keV_plot_z4[i] = r2_ave_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r2_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    
    r3_ave_cdm_plot_z4[i] = r3_ave_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r3_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_3keV_plot_z4[i] = r3_ave_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r3_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_4keV_plot_z4[i] = r3_ave_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r3_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_6keV_plot_z4[i] = r3_ave_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r3_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_9keV_plot_z4[i] = r3_ave_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r3_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    r4_ave_cdm_plot_z4[i] = r4_ave_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r4_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_3keV_plot_z4[i] = r4_ave_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r4_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_4keV_plot_z4[i] = r4_ave_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r4_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_6keV_plot_z4[i] = r4_ave_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r4_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_9keV_plot_z4[i] = r4_ave_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + r4_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_plot_z4[i] = ave_r1_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r1_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_3keV_plot_z4[i] = ave_r1_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r1_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_4keV_plot_z4[i] = ave_r1_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r1_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_6keV_plot_z4[i] = ave_r1_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r1_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_9keV_plot_z4[i] = ave_r1_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r1_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r2_cdm_plot_z4[i] = ave_r2_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r2_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_3keV_plot_z4[i] = ave_r2_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r2_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_4keV_plot_z4[i] = ave_r2_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r2_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_6keV_plot_z4[i] = ave_r2_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r2_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_9keV_plot_z4[i] = ave_r2_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r2_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r3_cdm_plot_z4[i] = ave_r3_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r3_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_3keV_plot_z4[i] = ave_r3_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r3_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_4keV_plot_z4[i] = ave_r3_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r3_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_6keV_plot_z4[i] = ave_r3_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r3_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_9keV_plot_z4[i] = ave_r3_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r3_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r4_cdm_plot_z4[i] = ave_r4_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r4_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_3keV_plot_z4[i] = ave_r4_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r4_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_4keV_plot_z4[i] = ave_r4_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r4_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_6keV_plot_z4[i] = ave_r4_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r4_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_9keV_plot_z4[i] = ave_r4_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r4_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r5_cdm_plot_z4[i] = ave_r5_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r5_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_3keV_plot_z4[i] = ave_r5_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r5_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_4keV_plot_z4[i] = ave_r5_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r5_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_6keV_plot_z4[i] = ave_r5_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r5_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_9keV_plot_z4[i] = ave_r5_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r5_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r6_cdm_plot_z4[i] = ave_r6_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r6_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_3keV_plot_z4[i] = ave_r6_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r6_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_4keV_plot_z4[i] = ave_r6_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r6_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_6keV_plot_z4[i] = ave_r6_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r6_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_9keV_plot_z4[i] = ave_r6_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r6_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r7_cdm_plot_z4[i] = ave_r7_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r7_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_3keV_plot_z4[i] = ave_r7_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r7_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_4keV_plot_z4[i] = ave_r7_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r7_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_6keV_plot_z4[i] = ave_r7_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r7_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_9keV_plot_z4[i] = ave_r7_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r7_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r8_cdm_plot_z4[i] = ave_r8_cdm.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r8_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_3keV_plot_z4[i] = ave_r8_3keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r8_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_4keV_plot_z4[i] = ave_r8_4keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r8_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_6keV_plot_z4[i] = ave_r8_6keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r8_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_9keV_plot_z4[i] = ave_r8_9keV.FluxP3D_lya_Mpc(z4, k_array[i], 0.1) + ave_r8_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    # next redshift
    # average
    ave_ave_cdm_plot_z5[i] = ave_ave_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_3keV_plot_z5[i] = ave_ave_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_4keV_plot_z5[i] = ave_ave_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_6keV_plot_z5[i] = ave_ave_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_9keV_plot_z5[i] = ave_ave_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_plot_z5[i] = r1_ave_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r1_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_3keV_plot_z5[i] = r1_ave_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r1_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_4keV_plot_z5[i] = r1_ave_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r1_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_6keV_plot_z5[i] = r1_ave_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r1_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_9keV_plot_z5[i] = r1_ave_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r1_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    
    r2_ave_cdm_plot_z5[i] = r2_ave_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r2_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_3keV_plot_z5[i] = r2_ave_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r2_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_4keV_plot_z5[i] = r2_ave_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r2_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_6keV_plot_z5[i] = r2_ave_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r2_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_9keV_plot_z5[i] = r2_ave_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r2_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    
    r3_ave_cdm_plot_z5[i] = r3_ave_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r3_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_3keV_plot_z5[i] = r3_ave_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r3_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_4keV_plot_z5[i] = r3_ave_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r3_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_6keV_plot_z5[i] = r3_ave_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r3_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_9keV_plot_z5[i] = r3_ave_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r3_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    r4_ave_cdm_plot_z5[i] = r4_ave_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r4_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_3keV_plot_z5[i] = r4_ave_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r4_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_4keV_plot_z5[i] = r4_ave_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r4_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_6keV_plot_z5[i] = r4_ave_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r4_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_9keV_plot_z5[i] = r4_ave_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + r4_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_plot_z5[i] = ave_r1_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r1_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_3keV_plot_z5[i] = ave_r1_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r1_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_4keV_plot_z5[i] = ave_r1_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r1_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_6keV_plot_z5[i] = ave_r1_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r1_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_9keV_plot_z5[i] = ave_r1_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r1_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r2_cdm_plot_z5[i] = ave_r2_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r2_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_3keV_plot_z5[i] = ave_r2_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r2_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_4keV_plot_z5[i] = ave_r2_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r2_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_6keV_plot_z5[i] = ave_r2_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r2_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_9keV_plot_z5[i] = ave_r2_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r2_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r3_cdm_plot_z5[i] = ave_r3_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r3_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_3keV_plot_z5[i] = ave_r3_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r3_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_4keV_plot_z5[i] = ave_r3_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r3_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_6keV_plot_z5[i] = ave_r3_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r3_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_9keV_plot_z5[i] = ave_r3_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r3_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r4_cdm_plot_z5[i] = ave_r4_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r4_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_3keV_plot_z5[i] = ave_r4_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r4_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_4keV_plot_z5[i] = ave_r4_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r4_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_6keV_plot_z5[i] = ave_r4_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r4_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_9keV_plot_z5[i] = ave_r4_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r4_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1) 
    
    ave_r5_cdm_plot_z5[i] = ave_r5_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r5_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_3keV_plot_z5[i] = ave_r5_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r5_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_4keV_plot_z5[i] = ave_r5_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r5_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_6keV_plot_z5[i] = ave_r5_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r5_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_9keV_plot_z5[i] = ave_r5_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r5_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r6_cdm_plot_z5[i] = ave_r6_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r6_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_3keV_plot_z5[i] = ave_r6_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r6_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_4keV_plot_z5[i] = ave_r6_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r6_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_6keV_plot_z5[i] = ave_r6_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r6_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_9keV_plot_z5[i] = ave_r6_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r6_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r7_cdm_plot_z5[i] = ave_r7_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r7_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_3keV_plot_z5[i] = ave_r7_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r7_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_4keV_plot_z5[i] = ave_r7_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r7_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_6keV_plot_z5[i] = ave_r7_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r7_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_9keV_plot_z5[i] = ave_r7_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r7_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r8_cdm_plot_z5[i] = ave_r8_cdm.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r8_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_3keV_plot_z5[i] = ave_r8_3keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r8_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_4keV_plot_z5[i] = ave_r8_4keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r8_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_6keV_plot_z5[i] = ave_r8_6keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r8_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_9keV_plot_z5[i] = ave_r8_9keV.FluxP3D_lya_Mpc(z5, k_array[i], 0.1) + ave_r8_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)   


for i in range(0,len(k_array)):
    # frist redshift
    # average
    ave_ave_cdm_reio_z1[i] = ave_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_3keV_reio_z1[i] = ave_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_4keV_reio_z1[i] = ave_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_6keV_reio_z1[i] = ave_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_ave_9keV_reio_z1[i] = ave_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_reio_z1[i] = r1_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_3keV_reio_z1[i] = r1_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_4keV_reio_z1[i] = r1_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_6keV_reio_z1[i] = r1_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r1_ave_9keV_reio_z1[i] = r1_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    
    r2_ave_cdm_reio_z1[i] = r2_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_3keV_reio_z1[i] = r2_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_4keV_reio_z1[i] = r2_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_6keV_reio_z1[i] = r2_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r2_ave_9keV_reio_z1[i] = r2_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    
    r3_ave_cdm_reio_z1[i] = r3_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_3keV_reio_z1[i] = r3_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_4keV_reio_z1[i] = r3_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_6keV_reio_z1[i] = r3_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r3_ave_9keV_reio_z1[i] = r3_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    r4_ave_cdm_reio_z1[i] = r4_ave_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_3keV_reio_z1[i] = r4_ave_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_4keV_reio_z1[i] = r4_ave_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_6keV_reio_z1[i] = r4_ave_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    r4_ave_9keV_reio_z1[i] = r4_ave_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_reio_z1[i] =  ave_r1_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_3keV_reio_z1[i] = ave_r1_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_4keV_reio_z1[i] = ave_r1_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_6keV_reio_z1[i] = ave_r1_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r1_9keV_reio_z1[i] = ave_r1_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r2_cdm_reio_z1[i] = ave_r2_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_3keV_reio_z1[i] = ave_r2_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_4keV_reio_z1[i] = ave_r2_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_6keV_reio_z1[i] = ave_r2_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r2_9keV_reio_z1[i] = ave_r2_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r3_cdm_reio_z1[i] = ave_r3_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_3keV_reio_z1[i] = ave_r3_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_4keV_reio_z1[i] = ave_r3_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_6keV_reio_z1[i] = ave_r3_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r3_9keV_reio_z1[i] = ave_r3_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r4_cdm_reio_z1[i] = ave_r4_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_3keV_reio_z1[i] = ave_r4_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_4keV_reio_z1[i] = ave_r4_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_6keV_reio_z1[i] = ave_r4_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r4_9keV_reio_z1[i] = ave_r4_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r5_cdm_reio_z1[i] = ave_r5_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_3keV_reio_z1[i] = ave_r5_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_4keV_reio_z1[i] = ave_r5_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_6keV_reio_z1[i] = ave_r5_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r5_9keV_reio_z1[i] = ave_r5_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r6_cdm_reio_z1[i] = ave_r6_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_3keV_reio_z1[i] = ave_r6_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_4keV_reio_z1[i] = ave_r6_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_6keV_reio_z1[i] = ave_r6_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r6_9keV_reio_z1[i] = ave_r6_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r7_cdm_reio_z1[i] = ave_r7_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_3keV_reio_z1[i] = ave_r7_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_4keV_reio_z1[i] = ave_r7_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_6keV_reio_z1[i] = ave_r7_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r7_9keV_reio_z1[i] = ave_r7_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)

    ave_r8_cdm_reio_z1[i] = ave_r8_cdm.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_3keV_reio_z1[i] = ave_r8_3keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_4keV_reio_z1[i] = ave_r8_4keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_6keV_reio_z1[i] = ave_r8_6keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)
    ave_r8_9keV_reio_z1[i] = ave_r8_9keV.FluxP3D_reio_Mpc(z1, k_array[i], 0.1)


# z2
for i in range(0,len(k_array)):
    ave_ave_cdm_reio_z2[i] = ave_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_3keV_reio_z2[i] = ave_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_4keV_reio_z2[i] = ave_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_6keV_reio_z2[i] = ave_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_ave_9keV_reio_z2[i] = ave_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_reio_z2[i] = r1_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_3keV_reio_z2[i] = r1_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_4keV_reio_z2[i] = r1_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_6keV_reio_z2[i] = r1_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r1_ave_9keV_reio_z2[i] = r1_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    
    r2_ave_cdm_reio_z2[i] = r2_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_3keV_reio_z2[i] = r2_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_4keV_reio_z2[i] = r2_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_6keV_reio_z2[i] = r2_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r2_ave_9keV_reio_z2[i] = r2_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    
    r3_ave_cdm_reio_z2[i] = r3_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_3keV_reio_z2[i] = r3_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_4keV_reio_z2[i] = r3_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_6keV_reio_z2[i] = r3_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r3_ave_9keV_reio_z2[i] = r3_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    r4_ave_cdm_reio_z2[i] = r4_ave_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_3keV_reio_z2[i] = r4_ave_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_4keV_reio_z2[i] = r4_ave_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_6keV_reio_z2[i] = r4_ave_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    r4_ave_9keV_reio_z2[i] = r4_ave_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_reio_z2[i] =  ave_r1_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_3keV_reio_z2[i] = ave_r1_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_4keV_reio_z2[i] = ave_r1_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_6keV_reio_z2[i] = ave_r1_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r1_9keV_reio_z2[i] = ave_r1_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r2_cdm_reio_z2[i] = ave_r2_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_3keV_reio_z2[i] = ave_r2_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_4keV_reio_z2[i] = ave_r2_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_6keV_reio_z2[i] = ave_r2_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r2_9keV_reio_z2[i] = ave_r2_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r3_cdm_reio_z2[i] = ave_r3_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_3keV_reio_z2[i] = ave_r3_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_4keV_reio_z2[i] = ave_r3_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_6keV_reio_z2[i] = ave_r3_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r3_9keV_reio_z2[i] = ave_r3_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r4_cdm_reio_z2[i] = ave_r4_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_3keV_reio_z2[i] = ave_r4_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_4keV_reio_z2[i] = ave_r4_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_6keV_reio_z2[i] = ave_r4_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r4_9keV_reio_z2[i] = ave_r4_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r5_cdm_reio_z2[i] = ave_r5_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_3keV_reio_z2[i] = ave_r5_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_4keV_reio_z2[i] = ave_r5_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_6keV_reio_z2[i] = ave_r5_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r5_9keV_reio_z2[i] = ave_r5_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r6_cdm_reio_z2[i] = ave_r6_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_3keV_reio_z2[i] = ave_r6_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_4keV_reio_z2[i] = ave_r6_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_6keV_reio_z2[i] = ave_r6_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r6_9keV_reio_z2[i] = ave_r6_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r7_cdm_reio_z2[i] = ave_r7_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_3keV_reio_z2[i] = ave_r7_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_4keV_reio_z2[i] = ave_r7_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_6keV_reio_z2[i] = ave_r7_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r7_9keV_reio_z2[i] = ave_r7_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

    ave_r8_cdm_reio_z2[i] = ave_r8_cdm.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_3keV_reio_z2[i] = ave_r8_3keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_4keV_reio_z2[i] = ave_r8_4keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_6keV_reio_z2[i] = ave_r8_6keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)
    ave_r8_9keV_reio_z2[i] = ave_r8_9keV.FluxP3D_reio_Mpc(z2, k_array[i], 0.1)

# z3

for i in range(0,len(k_array)):
    ave_ave_cdm_reio_z3[i] = ave_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_3keV_reio_z3[i] = ave_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_4keV_reio_z3[i] = ave_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_6keV_reio_z3[i] = ave_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_ave_9keV_reio_z3[i] = ave_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_reio_z3[i] = r1_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_3keV_reio_z3[i] = r1_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_4keV_reio_z3[i] = r1_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_6keV_reio_z3[i] = r1_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r1_ave_9keV_reio_z3[i] = r1_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    
    r2_ave_cdm_reio_z3[i] = r2_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_3keV_reio_z3[i] = r2_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_4keV_reio_z3[i] = r2_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_6keV_reio_z3[i] = r2_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r2_ave_9keV_reio_z3[i] = r2_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    
    r3_ave_cdm_reio_z3[i] = r3_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_3keV_reio_z3[i] = r3_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_4keV_reio_z3[i] = r3_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_6keV_reio_z3[i] = r3_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r3_ave_9keV_reio_z3[i] = r3_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    r4_ave_cdm_reio_z3[i] = r4_ave_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_3keV_reio_z3[i] = r4_ave_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_4keV_reio_z3[i] = r4_ave_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_6keV_reio_z3[i] = r4_ave_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    r4_ave_9keV_reio_z3[i] = r4_ave_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_reio_z3[i] =  ave_r1_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_3keV_reio_z3[i] = ave_r1_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_4keV_reio_z3[i] = ave_r1_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_6keV_reio_z3[i] = ave_r1_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r1_9keV_reio_z3[i] = ave_r1_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r2_cdm_reio_z3[i] = ave_r2_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_3keV_reio_z3[i] = ave_r2_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_4keV_reio_z3[i] = ave_r2_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_6keV_reio_z3[i] = ave_r2_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r2_9keV_reio_z3[i] = ave_r2_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r3_cdm_reio_z3[i] = ave_r3_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_3keV_reio_z3[i] = ave_r3_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_4keV_reio_z3[i] = ave_r3_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_6keV_reio_z3[i] = ave_r3_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r3_9keV_reio_z3[i] = ave_r3_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r4_cdm_reio_z3[i] = ave_r4_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_3keV_reio_z3[i] = ave_r4_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_4keV_reio_z3[i] = ave_r4_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_6keV_reio_z3[i] = ave_r4_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r4_9keV_reio_z3[i] = ave_r4_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r5_cdm_reio_z3[i] = ave_r5_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_3keV_reio_z3[i] = ave_r5_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_4keV_reio_z3[i] = ave_r5_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_6keV_reio_z3[i] = ave_r5_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r5_9keV_reio_z3[i] = ave_r5_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r6_cdm_reio_z3[i] = ave_r6_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_3keV_reio_z3[i] = ave_r6_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_4keV_reio_z3[i] = ave_r6_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_6keV_reio_z3[i] = ave_r6_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r6_9keV_reio_z3[i] = ave_r6_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r7_cdm_reio_z3[i] = ave_r7_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_3keV_reio_z3[i] = ave_r7_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_4keV_reio_z3[i] = ave_r7_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_6keV_reio_z3[i] = ave_r7_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r7_9keV_reio_z3[i] = ave_r7_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

    ave_r8_cdm_reio_z3[i] = ave_r8_cdm.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_3keV_reio_z3[i] = ave_r8_3keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_4keV_reio_z3[i] = ave_r8_4keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_6keV_reio_z3[i] = ave_r8_6keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)
    ave_r8_9keV_reio_z3[i] = ave_r8_9keV.FluxP3D_reio_Mpc(z3, k_array[i], 0.1)

# z4
for i in range(0,len(k_array)):
    ave_ave_cdm_reio_z4[i] = ave_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_3keV_reio_z4[i] = ave_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_4keV_reio_z4[i] = ave_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_6keV_reio_z4[i] = ave_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_ave_9keV_reio_z4[i] = ave_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_reio_z4[i] = r1_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_3keV_reio_z4[i] = r1_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_4keV_reio_z4[i] = r1_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_6keV_reio_z4[i] = r1_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r1_ave_9keV_reio_z4[i] = r1_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    
    r2_ave_cdm_reio_z4[i] = r2_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_3keV_reio_z4[i] = r2_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_4keV_reio_z4[i] = r2_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_6keV_reio_z4[i] = r2_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r2_ave_9keV_reio_z4[i] = r2_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    
    r3_ave_cdm_reio_z4[i] = r3_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_3keV_reio_z4[i] = r3_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_4keV_reio_z4[i] = r3_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_6keV_reio_z4[i] = r3_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r3_ave_9keV_reio_z4[i] = r3_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    r4_ave_cdm_reio_z4[i] = r4_ave_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_3keV_reio_z4[i] = r4_ave_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_4keV_reio_z4[i] = r4_ave_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_6keV_reio_z4[i] = r4_ave_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    r4_ave_9keV_reio_z4[i] = r4_ave_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_reio_z4[i] =  ave_r1_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_3keV_reio_z4[i] = ave_r1_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_4keV_reio_z4[i] = ave_r1_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_6keV_reio_z4[i] = ave_r1_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r1_9keV_reio_z4[i] = ave_r1_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r2_cdm_reio_z4[i] = ave_r2_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_3keV_reio_z4[i] = ave_r2_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_4keV_reio_z4[i] = ave_r2_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_6keV_reio_z4[i] = ave_r2_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r2_9keV_reio_z4[i] = ave_r2_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r3_cdm_reio_z4[i] = ave_r3_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_3keV_reio_z4[i] = ave_r3_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_4keV_reio_z4[i] = ave_r3_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_6keV_reio_z4[i] = ave_r3_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r3_9keV_reio_z4[i] = ave_r3_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r4_cdm_reio_z4[i] = ave_r4_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_3keV_reio_z4[i] = ave_r4_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_4keV_reio_z4[i] = ave_r4_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_6keV_reio_z4[i] = ave_r4_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r4_9keV_reio_z4[i] = ave_r4_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r5_cdm_reio_z4[i] = ave_r5_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_3keV_reio_z4[i] = ave_r5_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_4keV_reio_z4[i] = ave_r5_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_6keV_reio_z4[i] = ave_r5_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r5_9keV_reio_z4[i] = ave_r5_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r6_cdm_reio_z4[i] = ave_r6_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_3keV_reio_z4[i] = ave_r6_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_4keV_reio_z4[i] = ave_r6_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_6keV_reio_z4[i] = ave_r6_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r6_9keV_reio_z4[i] = ave_r6_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r7_cdm_reio_z4[i] = ave_r7_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_3keV_reio_z4[i] = ave_r7_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_4keV_reio_z4[i] = ave_r7_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_6keV_reio_z4[i] = ave_r7_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r7_9keV_reio_z4[i] = ave_r7_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

    ave_r8_cdm_reio_z4[i] = ave_r8_cdm.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_3keV_reio_z4[i] = ave_r8_3keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_4keV_reio_z4[i] = ave_r8_4keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_6keV_reio_z4[i] = ave_r8_6keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)
    ave_r8_9keV_reio_z4[i] = ave_r8_9keV.FluxP3D_reio_Mpc(z4, k_array[i], 0.1)

#z5

for i in range(0,len(k_array)):
    # frist redshift
    # average
    ave_ave_cdm_reio_z5[i] = ave_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_3keV_reio_z5[i] = ave_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_4keV_reio_z5[i] = ave_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_6keV_reio_z5[i] = ave_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_ave_9keV_reio_z5[i] = ave_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    # 21cmfast realizations
    r1_ave_cdm_reio_z5[i] = r1_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_3keV_reio_z5[i] = r1_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_4keV_reio_z5[i] = r1_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_6keV_reio_z5[i] = r1_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r1_ave_9keV_reio_z5[i] = r1_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    
    r2_ave_cdm_reio_z5[i] = r2_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_3keV_reio_z5[i] = r2_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_4keV_reio_z5[i] = r2_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_6keV_reio_z5[i] = r2_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r2_ave_9keV_reio_z5[i] = r2_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    
    r3_ave_cdm_reio_z5[i] = r3_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_3keV_reio_z5[i] = r3_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_4keV_reio_z5[i] = r3_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_6keV_reio_z5[i] = r3_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r3_ave_9keV_reio_z5[i] = r3_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    r4_ave_cdm_reio_z5[i] = r4_ave_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_3keV_reio_z5[i] = r4_ave_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_4keV_reio_z5[i] = r4_ave_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_6keV_reio_z5[i] = r4_ave_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    r4_ave_9keV_reio_z5[i] = r4_ave_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    # gadget realizations
    ave_r1_cdm_reio_z5[i] =  ave_r1_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_3keV_reio_z5[i] = ave_r1_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_4keV_reio_z5[i] = ave_r1_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_6keV_reio_z5[i] = ave_r1_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r1_9keV_reio_z5[i] = ave_r1_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r2_cdm_reio_z5[i] = ave_r2_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_3keV_reio_z5[i] = ave_r2_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_4keV_reio_z5[i] = ave_r2_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_6keV_reio_z5[i] = ave_r2_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r2_9keV_reio_z5[i] = ave_r2_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r3_cdm_reio_z5[i] = ave_r3_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_3keV_reio_z5[i] = ave_r3_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_4keV_reio_z5[i] = ave_r3_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_6keV_reio_z5[i] = ave_r3_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r3_9keV_reio_z5[i] = ave_r3_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r4_cdm_reio_z5[i] = ave_r4_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_3keV_reio_z5[i] = ave_r4_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_4keV_reio_z5[i] = ave_r4_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_6keV_reio_z5[i] = ave_r4_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r4_9keV_reio_z5[i] = ave_r4_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r5_cdm_reio_z5[i] = ave_r5_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_3keV_reio_z5[i] = ave_r5_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_4keV_reio_z5[i] = ave_r5_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_6keV_reio_z5[i] = ave_r5_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r5_9keV_reio_z5[i] = ave_r5_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r6_cdm_reio_z5[i] = ave_r6_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_3keV_reio_z5[i] = ave_r6_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_4keV_reio_z5[i] = ave_r6_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_6keV_reio_z5[i] = ave_r6_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r6_9keV_reio_z5[i] = ave_r6_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r7_cdm_reio_z5[i] = ave_r7_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_3keV_reio_z5[i] = ave_r7_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_4keV_reio_z5[i] = ave_r7_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_6keV_reio_z5[i] = ave_r7_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r7_9keV_reio_z5[i] = ave_r7_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)

    ave_r8_cdm_reio_z5[i] = ave_r8_cdm.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_3keV_reio_z5[i] = ave_r8_3keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_4keV_reio_z5[i] = ave_r8_4keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_6keV_reio_z5[i] = ave_r8_6keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)
    ave_r8_9keV_reio_z5[i] = ave_r8_9keV.FluxP3D_reio_Mpc(z5, k_array[i], 0.1)


# now we have all the data, so we can actually compute the errorbar
ave_ave_cdm_error_z1 = sample_variance(ave_ave_cdm_plot_z1, r1_ave_cdm_plot_z1, r2_ave_cdm_plot_z1, r3_ave_cdm_plot_z1, r4_ave_cdm_plot_z1, ave_r1_cdm_plot_z1, ave_r2_cdm_plot_z1, ave_r3_cdm_plot_z1, ave_r4_cdm_plot_z1, ave_r5_cdm_plot_z1, ave_r6_cdm_plot_z1, ave_r7_cdm_plot_z1, ave_r8_cdm_plot_z1)
ave_ave_3keV_error_z1 = sample_variance(ave_ave_3keV_plot_z1, r1_ave_3keV_plot_z1, r2_ave_3keV_plot_z1, r3_ave_3keV_plot_z1, r4_ave_3keV_plot_z1, ave_r1_3keV_plot_z1, ave_r2_3keV_plot_z1, ave_r3_3keV_plot_z1, ave_r4_3keV_plot_z1, ave_r5_3keV_plot_z1, ave_r6_3keV_plot_z1, ave_r7_3keV_plot_z1, ave_r8_3keV_plot_z1)
ave_ave_4keV_error_z1 = sample_variance(ave_ave_4keV_plot_z1, r1_ave_4keV_plot_z1, r2_ave_4keV_plot_z1, r3_ave_4keV_plot_z1, r4_ave_4keV_plot_z1, ave_r1_4keV_plot_z1, ave_r2_4keV_plot_z1, ave_r3_4keV_plot_z1, ave_r4_4keV_plot_z1, ave_r5_4keV_plot_z1, ave_r6_4keV_plot_z1, ave_r7_4keV_plot_z1, ave_r8_4keV_plot_z1)
ave_ave_6keV_error_z1 = sample_variance(ave_ave_6keV_plot_z1, r1_ave_6keV_plot_z1, r2_ave_6keV_plot_z1, r3_ave_6keV_plot_z1, r4_ave_6keV_plot_z1, ave_r1_6keV_plot_z1, ave_r2_6keV_plot_z1, ave_r3_6keV_plot_z1, ave_r4_6keV_plot_z1, ave_r5_6keV_plot_z1, ave_r6_6keV_plot_z1, ave_r7_6keV_plot_z1, ave_r8_6keV_plot_z1)
ave_ave_9keV_error_z1 = sample_variance(ave_ave_9keV_plot_z1, r1_ave_9keV_plot_z1, r2_ave_9keV_plot_z1, r3_ave_9keV_plot_z1, r4_ave_9keV_plot_z1, ave_r1_9keV_plot_z1, ave_r2_9keV_plot_z1, ave_r3_9keV_plot_z1, ave_r4_9keV_plot_z1, ave_r5_9keV_plot_z1, ave_r6_9keV_plot_z1, ave_r7_9keV_plot_z1, ave_r8_9keV_plot_z1)
# z2
ave_ave_cdm_error_z2 = sample_variance(ave_ave_cdm_plot_z2, r1_ave_cdm_plot_z2, r2_ave_cdm_plot_z2, r3_ave_cdm_plot_z2, r4_ave_cdm_plot_z2, ave_r1_cdm_plot_z2, ave_r2_cdm_plot_z2, ave_r3_cdm_plot_z2, ave_r4_cdm_plot_z2, ave_r5_cdm_plot_z2, ave_r6_cdm_plot_z2, ave_r7_cdm_plot_z2, ave_r8_cdm_plot_z2)
ave_ave_3keV_error_z2 = sample_variance(ave_ave_3keV_plot_z2, r1_ave_3keV_plot_z2, r2_ave_3keV_plot_z2, r3_ave_3keV_plot_z2, r4_ave_3keV_plot_z2, ave_r1_3keV_plot_z2, ave_r2_3keV_plot_z2, ave_r3_3keV_plot_z2, ave_r4_3keV_plot_z2, ave_r5_3keV_plot_z2, ave_r6_3keV_plot_z2, ave_r7_3keV_plot_z2, ave_r8_3keV_plot_z2)
ave_ave_4keV_error_z2 = sample_variance(ave_ave_4keV_plot_z2, r1_ave_4keV_plot_z2, r2_ave_4keV_plot_z2, r3_ave_4keV_plot_z2, r4_ave_4keV_plot_z2, ave_r1_4keV_plot_z2, ave_r2_4keV_plot_z2, ave_r3_4keV_plot_z2, ave_r4_4keV_plot_z2, ave_r5_4keV_plot_z2, ave_r6_4keV_plot_z2, ave_r7_4keV_plot_z2, ave_r8_4keV_plot_z2)
ave_ave_6keV_error_z2 = sample_variance(ave_ave_6keV_plot_z2, r1_ave_6keV_plot_z2, r2_ave_6keV_plot_z2, r3_ave_6keV_plot_z2, r4_ave_6keV_plot_z2, ave_r1_6keV_plot_z2, ave_r2_6keV_plot_z2, ave_r3_6keV_plot_z2, ave_r4_6keV_plot_z2, ave_r5_6keV_plot_z2, ave_r6_6keV_plot_z2, ave_r7_6keV_plot_z2, ave_r8_6keV_plot_z2)
ave_ave_9keV_error_z2 = sample_variance(ave_ave_9keV_plot_z2, r1_ave_9keV_plot_z2, r2_ave_9keV_plot_z2, r3_ave_9keV_plot_z2, r4_ave_9keV_plot_z2, ave_r1_9keV_plot_z2, ave_r2_9keV_plot_z2, ave_r3_9keV_plot_z2, ave_r4_9keV_plot_z2, ave_r5_9keV_plot_z2, ave_r6_9keV_plot_z2, ave_r7_9keV_plot_z2, ave_r8_9keV_plot_z2)
# z3
ave_ave_cdm_error_z3 = sample_variance(ave_ave_cdm_plot_z3, r1_ave_cdm_plot_z3, r2_ave_cdm_plot_z3, r3_ave_cdm_plot_z3, r4_ave_cdm_plot_z3, ave_r1_cdm_plot_z3, ave_r2_cdm_plot_z3, ave_r3_cdm_plot_z3, ave_r4_cdm_plot_z3, ave_r5_cdm_plot_z3, ave_r6_cdm_plot_z3, ave_r7_cdm_plot_z3, ave_r8_cdm_plot_z3)
ave_ave_3keV_error_z3 = sample_variance(ave_ave_3keV_plot_z3, r1_ave_3keV_plot_z3, r2_ave_3keV_plot_z3, r3_ave_3keV_plot_z3, r4_ave_3keV_plot_z3, ave_r1_3keV_plot_z3, ave_r2_3keV_plot_z3, ave_r3_3keV_plot_z3, ave_r4_3keV_plot_z3, ave_r5_3keV_plot_z3, ave_r6_3keV_plot_z3, ave_r7_3keV_plot_z3, ave_r8_3keV_plot_z3)
ave_ave_4keV_error_z3 = sample_variance(ave_ave_4keV_plot_z3, r1_ave_4keV_plot_z3, r2_ave_4keV_plot_z3, r3_ave_4keV_plot_z3, r4_ave_4keV_plot_z3, ave_r1_4keV_plot_z3, ave_r2_4keV_plot_z3, ave_r3_4keV_plot_z3, ave_r4_4keV_plot_z3, ave_r5_4keV_plot_z3, ave_r6_4keV_plot_z3, ave_r7_4keV_plot_z3, ave_r8_4keV_plot_z3)
ave_ave_6keV_error_z3 = sample_variance(ave_ave_6keV_plot_z3, r1_ave_6keV_plot_z3, r2_ave_6keV_plot_z3, r3_ave_6keV_plot_z3, r4_ave_6keV_plot_z3, ave_r1_6keV_plot_z3, ave_r2_6keV_plot_z3, ave_r3_6keV_plot_z3, ave_r4_6keV_plot_z3, ave_r5_6keV_plot_z3, ave_r6_6keV_plot_z3, ave_r7_6keV_plot_z3, ave_r8_6keV_plot_z3)
ave_ave_9keV_error_z3 = sample_variance(ave_ave_9keV_plot_z3, r1_ave_9keV_plot_z3, r2_ave_9keV_plot_z3, r3_ave_9keV_plot_z3, r4_ave_9keV_plot_z3, ave_r1_9keV_plot_z3, ave_r2_9keV_plot_z3, ave_r3_9keV_plot_z3, ave_r4_9keV_plot_z3, ave_r5_9keV_plot_z3, ave_r6_9keV_plot_z3, ave_r7_9keV_plot_z3, ave_r8_9keV_plot_z3)
# z4
ave_ave_cdm_error_z4 = sample_variance(ave_ave_cdm_plot_z4, r1_ave_cdm_plot_z4, r2_ave_cdm_plot_z4, r3_ave_cdm_plot_z4, r4_ave_cdm_plot_z4, ave_r1_cdm_plot_z4, ave_r2_cdm_plot_z4, ave_r3_cdm_plot_z4, ave_r4_cdm_plot_z4, ave_r5_cdm_plot_z4, ave_r6_cdm_plot_z4, ave_r7_cdm_plot_z4, ave_r8_cdm_plot_z4)
ave_ave_3keV_error_z4 = sample_variance(ave_ave_3keV_plot_z4, r1_ave_3keV_plot_z4, r2_ave_3keV_plot_z4, r3_ave_3keV_plot_z4, r4_ave_3keV_plot_z4, ave_r1_3keV_plot_z4, ave_r2_3keV_plot_z4, ave_r3_3keV_plot_z4, ave_r4_3keV_plot_z4, ave_r5_3keV_plot_z4, ave_r6_3keV_plot_z4, ave_r7_3keV_plot_z4, ave_r8_3keV_plot_z4)
ave_ave_4keV_error_z4 = sample_variance(ave_ave_4keV_plot_z4, r1_ave_4keV_plot_z4, r2_ave_4keV_plot_z4, r3_ave_4keV_plot_z4, r4_ave_4keV_plot_z4, ave_r1_4keV_plot_z4, ave_r2_4keV_plot_z4, ave_r3_4keV_plot_z4, ave_r4_4keV_plot_z4, ave_r5_4keV_plot_z4, ave_r6_4keV_plot_z4, ave_r7_4keV_plot_z4, ave_r8_4keV_plot_z4)
ave_ave_6keV_error_z4 = sample_variance(ave_ave_6keV_plot_z4, r1_ave_6keV_plot_z4, r2_ave_6keV_plot_z4, r3_ave_6keV_plot_z4, r4_ave_6keV_plot_z4, ave_r1_6keV_plot_z4, ave_r2_6keV_plot_z4, ave_r3_6keV_plot_z4, ave_r4_6keV_plot_z4, ave_r5_6keV_plot_z4, ave_r6_6keV_plot_z4, ave_r7_6keV_plot_z4, ave_r8_6keV_plot_z4)
ave_ave_9keV_error_z4 = sample_variance(ave_ave_9keV_plot_z4, r1_ave_9keV_plot_z4, r2_ave_9keV_plot_z4, r3_ave_9keV_plot_z4, r4_ave_9keV_plot_z4, ave_r1_9keV_plot_z4, ave_r2_9keV_plot_z4, ave_r3_9keV_plot_z4, ave_r4_9keV_plot_z4, ave_r5_9keV_plot_z4, ave_r6_9keV_plot_z4, ave_r7_9keV_plot_z4, ave_r8_9keV_plot_z4)
# z5
ave_ave_cdm_error_z5 = sample_variance(ave_ave_cdm_plot_z5, r1_ave_cdm_plot_z5, r2_ave_cdm_plot_z5, r3_ave_cdm_plot_z5, r4_ave_cdm_plot_z5, ave_r1_cdm_plot_z5, ave_r2_cdm_plot_z5, ave_r3_cdm_plot_z5, ave_r4_cdm_plot_z5, ave_r5_cdm_plot_z5, ave_r6_cdm_plot_z5, ave_r7_cdm_plot_z5, ave_r8_cdm_plot_z5)
ave_ave_3keV_error_z5 = sample_variance(ave_ave_3keV_plot_z5, r1_ave_3keV_plot_z5, r2_ave_3keV_plot_z5, r3_ave_3keV_plot_z5, r4_ave_3keV_plot_z5, ave_r1_3keV_plot_z5, ave_r2_3keV_plot_z5, ave_r3_3keV_plot_z5, ave_r4_3keV_plot_z5, ave_r5_3keV_plot_z5, ave_r6_3keV_plot_z5, ave_r7_3keV_plot_z5, ave_r8_3keV_plot_z5)
ave_ave_4keV_error_z5 = sample_variance(ave_ave_4keV_plot_z5, r1_ave_4keV_plot_z5, r2_ave_4keV_plot_z5, r3_ave_4keV_plot_z5, r4_ave_4keV_plot_z5, ave_r1_4keV_plot_z5, ave_r2_4keV_plot_z5, ave_r3_4keV_plot_z5, ave_r4_4keV_plot_z5, ave_r5_4keV_plot_z5, ave_r6_4keV_plot_z5, ave_r7_4keV_plot_z5, ave_r8_4keV_plot_z5)
ave_ave_6keV_error_z5 = sample_variance(ave_ave_6keV_plot_z5, r1_ave_6keV_plot_z5, r2_ave_6keV_plot_z5, r3_ave_6keV_plot_z5, r4_ave_6keV_plot_z5, ave_r1_6keV_plot_z5, ave_r2_6keV_plot_z5, ave_r3_6keV_plot_z5, ave_r4_6keV_plot_z5, ave_r5_6keV_plot_z5, ave_r6_6keV_plot_z5, ave_r7_6keV_plot_z5, ave_r8_6keV_plot_z5)
ave_ave_9keV_error_z5 = sample_variance(ave_ave_9keV_plot_z5, r1_ave_9keV_plot_z5, r2_ave_9keV_plot_z5, r3_ave_9keV_plot_z5, r4_ave_9keV_plot_z5, ave_r1_9keV_plot_z5, ave_r2_9keV_plot_z5, ave_r3_9keV_plot_z5, ave_r4_9keV_plot_z5, ave_r5_9keV_plot_z5, ave_r6_9keV_plot_z5, ave_r7_9keV_plot_z5, ave_r8_9keV_plot_z5)
# now if we want to do fractional difference let's use what we have done
# first the "mean" one
dif_ave_ave_cdm_z1 = frac_diff(ave_ave_cdm_plot_z1, ave_ave_cdm_reio_z1)
dif_ave_ave_3keV_z1 = frac_diff(ave_ave_3keV_plot_z1, ave_ave_3keV_reio_z1)
dif_ave_ave_4keV_z1 = frac_diff(ave_ave_4keV_plot_z1, ave_ave_4keV_reio_z1)
dif_ave_ave_6keV_z1 = frac_diff(ave_ave_6keV_plot_z1, ave_ave_6keV_reio_z1)
dif_ave_ave_9keV_z1 = frac_diff(ave_ave_9keV_plot_z1, ave_ave_9keV_reio_z1)
# now the realizations, starting with 21cmFAST 
dif_r1_ave_cdm_z1 = frac_diff(r1_ave_cdm_plot_z1, r1_ave_cdm_reio_z1)
dif_r1_ave_3keV_z1 = frac_diff(r1_ave_3keV_plot_z1, r1_ave_3keV_reio_z1)
dif_r1_ave_4keV_z1 = frac_diff(r1_ave_4keV_plot_z1, r1_ave_4keV_reio_z1)
dif_r1_ave_6keV_z1 = frac_diff(r1_ave_6keV_plot_z1, r1_ave_6keV_reio_z1)
dif_r1_ave_9keV_z1 = frac_diff(r1_ave_9keV_plot_z1, r1_ave_9keV_reio_z1)

dif_r2_ave_cdm_z1 = frac_diff(r2_ave_cdm_plot_z1, r2_ave_cdm_reio_z1)
dif_r2_ave_3keV_z1 = frac_diff(r2_ave_3keV_plot_z1, r2_ave_3keV_reio_z1)
dif_r2_ave_4keV_z1 = frac_diff(r2_ave_4keV_plot_z1, r2_ave_4keV_reio_z1)
dif_r2_ave_6keV_z1 = frac_diff(r2_ave_6keV_plot_z1, r2_ave_6keV_reio_z1)
dif_r2_ave_9keV_z1 = frac_diff(r2_ave_9keV_plot_z1, r2_ave_9keV_reio_z1)

dif_r3_ave_cdm_z1 = frac_diff(r3_ave_cdm_plot_z1, r3_ave_cdm_reio_z1)
dif_r3_ave_3keV_z1 = frac_diff(r3_ave_3keV_plot_z1, r3_ave_3keV_reio_z1)
dif_r3_ave_4keV_z1 = frac_diff(r3_ave_4keV_plot_z1, r3_ave_4keV_reio_z1)
dif_r3_ave_6keV_z1 = frac_diff(r3_ave_6keV_plot_z1, r3_ave_6keV_reio_z1)
dif_r3_ave_9keV_z1 = frac_diff(r3_ave_9keV_plot_z1, r3_ave_9keV_reio_z1)

dif_r4_ave_cdm_z1 = frac_diff(r4_ave_cdm_plot_z1, r4_ave_cdm_reio_z1)
dif_r4_ave_3keV_z1 = frac_diff(r4_ave_3keV_plot_z1, r4_ave_3keV_reio_z1)
dif_r4_ave_4keV_z1 = frac_diff(r4_ave_4keV_plot_z1, r4_ave_4keV_reio_z1)
dif_r4_ave_6keV_z1 = frac_diff(r4_ave_6keV_plot_z1, r4_ave_6keV_reio_z1)
dif_r4_ave_9keV_z1 = frac_diff(r4_ave_9keV_plot_z1, r4_ave_9keV_reio_z1)

dif_ave_r1_cdm_z1 = frac_diff(ave_r1_cdm_plot_z1, ave_r1_cdm_reio_z1)
dif_ave_r1_3keV_z1 = frac_diff(ave_r1_3keV_plot_z1, ave_r1_3keV_reio_z1)
dif_ave_r1_4keV_z1 = frac_diff(ave_r1_4keV_plot_z1, ave_r1_4keV_reio_z1)
dif_ave_r1_6keV_z1 = frac_diff(ave_r1_6keV_plot_z1, ave_r1_6keV_reio_z1)
dif_ave_r1_9keV_z1 = frac_diff(ave_r1_9keV_plot_z1, ave_r1_9keV_reio_z1)

dif_ave_r2_cdm_z1 = frac_diff(ave_r2_cdm_plot_z1, ave_r2_cdm_reio_z1)
dif_ave_r2_3keV_z1 = frac_diff(ave_r2_3keV_plot_z1, ave_r2_3keV_reio_z1)
dif_ave_r2_4keV_z1 = frac_diff(ave_r2_4keV_plot_z1, ave_r2_4keV_reio_z1)
dif_ave_r2_6keV_z1 = frac_diff(ave_r2_6keV_plot_z1, ave_r2_6keV_reio_z1)
dif_ave_r2_9keV_z1 = frac_diff(ave_r2_9keV_plot_z1, ave_r2_9keV_reio_z1)

dif_ave_r3_cdm_z1 = frac_diff(ave_r3_cdm_plot_z1, ave_r3_cdm_reio_z1)
dif_ave_r3_3keV_z1 = frac_diff(ave_r3_3keV_plot_z1, ave_r3_3keV_reio_z1)
dif_ave_r3_4keV_z1 = frac_diff(ave_r3_4keV_plot_z1, ave_r3_4keV_reio_z1)
dif_ave_r3_6keV_z1 = frac_diff(ave_r3_6keV_plot_z1, ave_r3_6keV_reio_z1)
dif_ave_r3_9keV_z1 = frac_diff(ave_r3_9keV_plot_z1, ave_r3_9keV_reio_z1)

dif_ave_r4_cdm_z1 = frac_diff(ave_r4_cdm_plot_z1, ave_r4_cdm_reio_z1)
dif_ave_r4_3keV_z1 = frac_diff(ave_r4_3keV_plot_z1, ave_r4_3keV_reio_z1)
dif_ave_r4_4keV_z1 = frac_diff(ave_r4_4keV_plot_z1, ave_r4_4keV_reio_z1)
dif_ave_r4_6keV_z1 = frac_diff(ave_r4_6keV_plot_z1, ave_r4_6keV_reio_z1)
dif_ave_r4_9keV_z1 = frac_diff(ave_r4_9keV_plot_z1, ave_r4_9keV_reio_z1)

dif_ave_r5_cdm_z1 = frac_diff(ave_r5_cdm_plot_z1, ave_r5_cdm_reio_z1)
dif_ave_r5_3keV_z1 = frac_diff(ave_r5_3keV_plot_z1, ave_r5_3keV_reio_z1)
dif_ave_r5_4keV_z1 = frac_diff(ave_r5_4keV_plot_z1, ave_r5_4keV_reio_z1)
dif_ave_r5_6keV_z1 = frac_diff(ave_r5_6keV_plot_z1, ave_r5_6keV_reio_z1)
dif_ave_r5_9keV_z1 = frac_diff(ave_r5_9keV_plot_z1, ave_r5_9keV_reio_z1)

dif_ave_r6_cdm_z1 = frac_diff(ave_r6_cdm_plot_z1, ave_r6_cdm_reio_z1)
dif_ave_r6_3keV_z1 = frac_diff(ave_r6_3keV_plot_z1, ave_r6_3keV_reio_z1)
dif_ave_r6_4keV_z1 = frac_diff(ave_r6_4keV_plot_z1, ave_r6_4keV_reio_z1)
dif_ave_r6_6keV_z1 = frac_diff(ave_r6_6keV_plot_z1, ave_r6_6keV_reio_z1)
dif_ave_r6_9keV_z1 = frac_diff(ave_r6_9keV_plot_z1, ave_r6_9keV_reio_z1)

dif_ave_r7_cdm_z1 = frac_diff(ave_r7_cdm_plot_z1, ave_r7_cdm_reio_z1)
dif_ave_r7_3keV_z1 = frac_diff(ave_r7_3keV_plot_z1, ave_r7_3keV_reio_z1)
dif_ave_r7_4keV_z1 = frac_diff(ave_r7_4keV_plot_z1, ave_r7_4keV_reio_z1)
dif_ave_r7_6keV_z1 = frac_diff(ave_r7_6keV_plot_z1, ave_r7_6keV_reio_z1)
dif_ave_r7_9keV_z1 = frac_diff(ave_r7_9keV_plot_z1, ave_r7_9keV_reio_z1)

dif_ave_r8_cdm_z1 = frac_diff(ave_r8_cdm_plot_z1, ave_r8_cdm_reio_z1)
dif_ave_r8_3keV_z1 = frac_diff(ave_r8_3keV_plot_z1, ave_r8_3keV_reio_z1)
dif_ave_r8_4keV_z1 = frac_diff(ave_r8_4keV_plot_z1, ave_r8_4keV_reio_z1)
dif_ave_r8_6keV_z1 = frac_diff(ave_r8_6keV_plot_z1, ave_r8_6keV_reio_z1)
dif_ave_r8_9keV_z1 = frac_diff(ave_r8_9keV_plot_z1, ave_r8_9keV_reio_z1)

#z2
# first the "mean" one
dif_ave_ave_cdm_z2 = frac_diff(ave_ave_cdm_plot_z2, ave_ave_cdm_reio_z2)
dif_ave_ave_3keV_z2 = frac_diff(ave_ave_3keV_plot_z2, ave_ave_3keV_reio_z2)
dif_ave_ave_4keV_z2 = frac_diff(ave_ave_4keV_plot_z2, ave_ave_4keV_reio_z2)
dif_ave_ave_6keV_z2 = frac_diff(ave_ave_6keV_plot_z2, ave_ave_6keV_reio_z2)
dif_ave_ave_9keV_z2 = frac_diff(ave_ave_9keV_plot_z2, ave_ave_9keV_reio_z2)
# now the realizations, starting with 21cmFAST 
dif_r1_ave_cdm_z2 = frac_diff(r1_ave_cdm_plot_z2, r1_ave_cdm_reio_z2)
dif_r1_ave_3keV_z2 = frac_diff(r1_ave_3keV_plot_z2, r1_ave_3keV_reio_z2)
dif_r1_ave_4keV_z2 = frac_diff(r1_ave_4keV_plot_z2, r1_ave_4keV_reio_z2)
dif_r1_ave_6keV_z2 = frac_diff(r1_ave_6keV_plot_z2, r1_ave_6keV_reio_z2)
dif_r1_ave_9keV_z2 = frac_diff(r1_ave_9keV_plot_z2, r1_ave_9keV_reio_z2)

dif_r2_ave_cdm_z2 = frac_diff(r2_ave_cdm_plot_z2, r2_ave_cdm_reio_z2)
dif_r2_ave_3keV_z2 = frac_diff(r2_ave_3keV_plot_z2, r2_ave_3keV_reio_z2)
dif_r2_ave_4keV_z2 = frac_diff(r2_ave_4keV_plot_z2, r2_ave_4keV_reio_z2)
dif_r2_ave_6keV_z2 = frac_diff(r2_ave_6keV_plot_z2, r2_ave_6keV_reio_z2)
dif_r2_ave_9keV_z2 = frac_diff(r2_ave_9keV_plot_z2, r2_ave_9keV_reio_z2)

dif_r3_ave_cdm_z2 = frac_diff(r3_ave_cdm_plot_z2, r3_ave_cdm_reio_z2)
dif_r3_ave_3keV_z2 = frac_diff(r3_ave_3keV_plot_z2, r3_ave_3keV_reio_z2)
dif_r3_ave_4keV_z2 = frac_diff(r3_ave_4keV_plot_z2, r3_ave_4keV_reio_z2)
dif_r3_ave_6keV_z2 = frac_diff(r3_ave_6keV_plot_z2, r3_ave_6keV_reio_z2)
dif_r3_ave_9keV_z2 = frac_diff(r3_ave_9keV_plot_z2, r3_ave_9keV_reio_z2)

dif_r4_ave_cdm_z2 = frac_diff(r4_ave_cdm_plot_z2, r4_ave_cdm_reio_z2)
dif_r4_ave_3keV_z2 = frac_diff(r4_ave_3keV_plot_z2, r4_ave_3keV_reio_z2)
dif_r4_ave_4keV_z2 = frac_diff(r4_ave_4keV_plot_z2, r4_ave_4keV_reio_z2)
dif_r4_ave_6keV_z2 = frac_diff(r4_ave_6keV_plot_z2, r4_ave_6keV_reio_z2)
dif_r4_ave_9keV_z2 = frac_diff(r4_ave_9keV_plot_z2, r4_ave_9keV_reio_z2)

dif_ave_r1_cdm_z2 = frac_diff(ave_r1_cdm_plot_z2, ave_r1_cdm_reio_z2)
dif_ave_r1_3keV_z2 = frac_diff(ave_r1_3keV_plot_z2, ave_r1_3keV_reio_z2)
dif_ave_r1_4keV_z2 = frac_diff(ave_r1_4keV_plot_z2, ave_r1_4keV_reio_z2)
dif_ave_r1_6keV_z2 = frac_diff(ave_r1_6keV_plot_z2, ave_r1_6keV_reio_z2)
dif_ave_r1_9keV_z2 = frac_diff(ave_r1_9keV_plot_z2, ave_r1_9keV_reio_z2)

dif_ave_r2_cdm_z2 = frac_diff(ave_r2_cdm_plot_z2, ave_r2_cdm_reio_z2)
dif_ave_r2_3keV_z2 = frac_diff(ave_r2_3keV_plot_z2, ave_r2_3keV_reio_z2)
dif_ave_r2_4keV_z2 = frac_diff(ave_r2_4keV_plot_z2, ave_r2_4keV_reio_z2)
dif_ave_r2_6keV_z2 = frac_diff(ave_r2_6keV_plot_z2, ave_r2_6keV_reio_z2)
dif_ave_r2_9keV_z2 = frac_diff(ave_r2_9keV_plot_z2, ave_r2_9keV_reio_z2)

dif_ave_r3_cdm_z2 = frac_diff(ave_r3_cdm_plot_z2, ave_r3_cdm_reio_z2)
dif_ave_r3_3keV_z2 = frac_diff(ave_r3_3keV_plot_z2, ave_r3_3keV_reio_z2)
dif_ave_r3_4keV_z2 = frac_diff(ave_r3_4keV_plot_z2, ave_r3_4keV_reio_z2)
dif_ave_r3_6keV_z2 = frac_diff(ave_r3_6keV_plot_z2, ave_r3_6keV_reio_z2)
dif_ave_r3_9keV_z2 = frac_diff(ave_r3_9keV_plot_z2, ave_r3_9keV_reio_z2)

dif_ave_r4_cdm_z2 = frac_diff(ave_r4_cdm_plot_z2, ave_r4_cdm_reio_z2)
dif_ave_r4_3keV_z2 = frac_diff(ave_r4_3keV_plot_z2, ave_r4_3keV_reio_z2)
dif_ave_r4_4keV_z2 = frac_diff(ave_r4_4keV_plot_z2, ave_r4_4keV_reio_z2)
dif_ave_r4_6keV_z2 = frac_diff(ave_r4_6keV_plot_z2, ave_r4_6keV_reio_z2)
dif_ave_r4_9keV_z2 = frac_diff(ave_r4_9keV_plot_z2, ave_r4_9keV_reio_z2)

dif_ave_r5_cdm_z2 = frac_diff(ave_r5_cdm_plot_z2, ave_r5_cdm_reio_z2)
dif_ave_r5_3keV_z2 = frac_diff(ave_r5_3keV_plot_z2, ave_r5_3keV_reio_z2)
dif_ave_r5_4keV_z2 = frac_diff(ave_r5_4keV_plot_z2, ave_r5_4keV_reio_z2)
dif_ave_r5_6keV_z2 = frac_diff(ave_r5_6keV_plot_z2, ave_r5_6keV_reio_z2)
dif_ave_r5_9keV_z2 = frac_diff(ave_r5_9keV_plot_z2, ave_r5_9keV_reio_z2)

dif_ave_r6_cdm_z2 = frac_diff(ave_r6_cdm_plot_z2, ave_r6_cdm_reio_z2)
dif_ave_r6_3keV_z2 = frac_diff(ave_r6_3keV_plot_z2, ave_r6_3keV_reio_z2)
dif_ave_r6_4keV_z2 = frac_diff(ave_r6_4keV_plot_z2, ave_r6_4keV_reio_z2)
dif_ave_r6_6keV_z2 = frac_diff(ave_r6_6keV_plot_z2, ave_r6_6keV_reio_z2)
dif_ave_r6_9keV_z2 = frac_diff(ave_r6_9keV_plot_z2, ave_r6_9keV_reio_z2)

dif_ave_r7_cdm_z2 = frac_diff(ave_r7_cdm_plot_z2, ave_r7_cdm_reio_z2)
dif_ave_r7_3keV_z2 = frac_diff(ave_r7_3keV_plot_z2, ave_r7_3keV_reio_z2)
dif_ave_r7_4keV_z2 = frac_diff(ave_r7_4keV_plot_z2, ave_r7_4keV_reio_z2)
dif_ave_r7_6keV_z2 = frac_diff(ave_r7_6keV_plot_z2, ave_r7_6keV_reio_z2)
dif_ave_r7_9keV_z2 = frac_diff(ave_r7_9keV_plot_z2, ave_r7_9keV_reio_z2)

dif_ave_r8_cdm_z2 = frac_diff(ave_r8_cdm_plot_z2, ave_r8_cdm_reio_z2)
dif_ave_r8_3keV_z2 = frac_diff(ave_r8_3keV_plot_z2, ave_r8_3keV_reio_z2)
dif_ave_r8_4keV_z2 = frac_diff(ave_r8_4keV_plot_z2, ave_r8_4keV_reio_z2)
dif_ave_r8_6keV_z2 = frac_diff(ave_r8_6keV_plot_z2, ave_r8_6keV_reio_z2)
dif_ave_r8_9keV_z2 = frac_diff(ave_r8_9keV_plot_z2, ave_r8_9keV_reio_z2)

# z3
# first the "mean" one
dif_ave_ave_cdm_z3 = frac_diff(ave_ave_cdm_plot_z3, ave_ave_cdm_reio_z3)
dif_ave_ave_3keV_z3 = frac_diff(ave_ave_3keV_plot_z3, ave_ave_3keV_reio_z3)
dif_ave_ave_4keV_z3 = frac_diff(ave_ave_4keV_plot_z3, ave_ave_4keV_reio_z3)
dif_ave_ave_6keV_z3 = frac_diff(ave_ave_6keV_plot_z3, ave_ave_6keV_reio_z3)
dif_ave_ave_9keV_z3 = frac_diff(ave_ave_9keV_plot_z3, ave_ave_9keV_reio_z3)
# now the realizations, starting with 21cmFAST 
dif_r1_ave_cdm_z3 = frac_diff(r1_ave_cdm_plot_z3, r1_ave_cdm_reio_z3)
dif_r1_ave_3keV_z3 = frac_diff(r1_ave_3keV_plot_z3, r1_ave_3keV_reio_z3)
dif_r1_ave_4keV_z3 = frac_diff(r1_ave_4keV_plot_z3, r1_ave_4keV_reio_z3)
dif_r1_ave_6keV_z3 = frac_diff(r1_ave_6keV_plot_z3, r1_ave_6keV_reio_z3)
dif_r1_ave_9keV_z3 = frac_diff(r1_ave_9keV_plot_z3, r1_ave_9keV_reio_z3)

dif_r2_ave_cdm_z3 = frac_diff(r2_ave_cdm_plot_z3, r2_ave_cdm_reio_z3)
dif_r2_ave_3keV_z3 = frac_diff(r2_ave_3keV_plot_z3, r2_ave_3keV_reio_z3)
dif_r2_ave_4keV_z3 = frac_diff(r2_ave_4keV_plot_z3, r2_ave_4keV_reio_z3)
dif_r2_ave_6keV_z3 = frac_diff(r2_ave_6keV_plot_z3, r2_ave_6keV_reio_z3)
dif_r2_ave_9keV_z3 = frac_diff(r2_ave_9keV_plot_z3, r2_ave_9keV_reio_z3)

dif_r3_ave_cdm_z3 = frac_diff(r3_ave_cdm_plot_z3, r3_ave_cdm_reio_z3)
dif_r3_ave_3keV_z3 = frac_diff(r3_ave_3keV_plot_z3, r3_ave_3keV_reio_z3)
dif_r3_ave_4keV_z3 = frac_diff(r3_ave_4keV_plot_z3, r3_ave_4keV_reio_z3)
dif_r3_ave_6keV_z3 = frac_diff(r3_ave_6keV_plot_z3, r3_ave_6keV_reio_z3)
dif_r3_ave_9keV_z3 = frac_diff(r3_ave_9keV_plot_z3, r3_ave_9keV_reio_z3)

dif_r4_ave_cdm_z3 = frac_diff(r4_ave_cdm_plot_z3, r4_ave_cdm_reio_z3)
dif_r4_ave_3keV_z3 = frac_diff(r4_ave_3keV_plot_z3, r4_ave_3keV_reio_z3)
dif_r4_ave_4keV_z3 = frac_diff(r4_ave_4keV_plot_z3, r4_ave_4keV_reio_z3)
dif_r4_ave_6keV_z3 = frac_diff(r4_ave_6keV_plot_z3, r4_ave_6keV_reio_z3)
dif_r4_ave_9keV_z3 = frac_diff(r4_ave_9keV_plot_z3, r4_ave_9keV_reio_z3)

dif_ave_r1_cdm_z3 = frac_diff(ave_r1_cdm_plot_z3, ave_r1_cdm_reio_z3)
dif_ave_r1_3keV_z3 = frac_diff(ave_r1_3keV_plot_z3, ave_r1_3keV_reio_z3)
dif_ave_r1_4keV_z3 = frac_diff(ave_r1_4keV_plot_z3, ave_r1_4keV_reio_z3)
dif_ave_r1_6keV_z3 = frac_diff(ave_r1_6keV_plot_z3, ave_r1_6keV_reio_z3)
dif_ave_r1_9keV_z3 = frac_diff(ave_r1_9keV_plot_z3, ave_r1_9keV_reio_z3)

dif_ave_r2_cdm_z3 = frac_diff(ave_r2_cdm_plot_z3, ave_r2_cdm_reio_z3)
dif_ave_r2_3keV_z3 = frac_diff(ave_r2_3keV_plot_z3, ave_r2_3keV_reio_z3)
dif_ave_r2_4keV_z3 = frac_diff(ave_r2_4keV_plot_z3, ave_r2_4keV_reio_z3)
dif_ave_r2_6keV_z3 = frac_diff(ave_r2_6keV_plot_z3, ave_r2_6keV_reio_z3)
dif_ave_r2_9keV_z3 = frac_diff(ave_r2_9keV_plot_z3, ave_r2_9keV_reio_z3)

dif_ave_r3_cdm_z3 = frac_diff(ave_r3_cdm_plot_z3, ave_r3_cdm_reio_z3)
dif_ave_r3_3keV_z3 = frac_diff(ave_r3_3keV_plot_z3, ave_r3_3keV_reio_z3)
dif_ave_r3_4keV_z3 = frac_diff(ave_r3_4keV_plot_z3, ave_r3_4keV_reio_z3)
dif_ave_r3_6keV_z3 = frac_diff(ave_r3_6keV_plot_z3, ave_r3_6keV_reio_z3)
dif_ave_r3_9keV_z3 = frac_diff(ave_r3_9keV_plot_z3, ave_r3_9keV_reio_z3)

dif_ave_r4_cdm_z3 = frac_diff(ave_r4_cdm_plot_z3, ave_r4_cdm_reio_z3)
dif_ave_r4_3keV_z3 = frac_diff(ave_r4_3keV_plot_z3, ave_r4_3keV_reio_z3)
dif_ave_r4_4keV_z3 = frac_diff(ave_r4_4keV_plot_z3, ave_r4_4keV_reio_z3)
dif_ave_r4_6keV_z3 = frac_diff(ave_r4_6keV_plot_z3, ave_r4_6keV_reio_z3)
dif_ave_r4_9keV_z3 = frac_diff(ave_r4_9keV_plot_z3, ave_r4_9keV_reio_z3)

dif_ave_r5_cdm_z3 = frac_diff(ave_r5_cdm_plot_z3, ave_r5_cdm_reio_z3)
dif_ave_r5_3keV_z3 = frac_diff(ave_r5_3keV_plot_z3, ave_r5_3keV_reio_z3)
dif_ave_r5_4keV_z3 = frac_diff(ave_r5_4keV_plot_z3, ave_r5_4keV_reio_z3)
dif_ave_r5_6keV_z3 = frac_diff(ave_r5_6keV_plot_z3, ave_r5_6keV_reio_z3)
dif_ave_r5_9keV_z3 = frac_diff(ave_r5_9keV_plot_z3, ave_r5_9keV_reio_z3)

dif_ave_r6_cdm_z3 = frac_diff(ave_r6_cdm_plot_z3, ave_r6_cdm_reio_z3)
dif_ave_r6_3keV_z3 = frac_diff(ave_r6_3keV_plot_z3, ave_r6_3keV_reio_z3)
dif_ave_r6_4keV_z3 = frac_diff(ave_r6_4keV_plot_z3, ave_r6_4keV_reio_z3)
dif_ave_r6_6keV_z3 = frac_diff(ave_r6_6keV_plot_z3, ave_r6_6keV_reio_z3)
dif_ave_r6_9keV_z3 = frac_diff(ave_r6_9keV_plot_z3, ave_r6_9keV_reio_z3)

dif_ave_r7_cdm_z3 = frac_diff(ave_r7_cdm_plot_z3, ave_r7_cdm_reio_z3)
dif_ave_r7_3keV_z3 = frac_diff(ave_r7_3keV_plot_z3, ave_r7_3keV_reio_z3)
dif_ave_r7_4keV_z3 = frac_diff(ave_r7_4keV_plot_z3, ave_r7_4keV_reio_z3)
dif_ave_r7_6keV_z3 = frac_diff(ave_r7_6keV_plot_z3, ave_r7_6keV_reio_z3)
dif_ave_r7_9keV_z3 = frac_diff(ave_r7_9keV_plot_z3, ave_r7_9keV_reio_z3)

dif_ave_r8_cdm_z3 = frac_diff(ave_r8_cdm_plot_z3, ave_r8_cdm_reio_z3)
dif_ave_r8_3keV_z3 = frac_diff(ave_r8_3keV_plot_z3, ave_r8_3keV_reio_z3)
dif_ave_r8_4keV_z3 = frac_diff(ave_r8_4keV_plot_z3, ave_r8_4keV_reio_z3)
dif_ave_r8_6keV_z3 = frac_diff(ave_r8_6keV_plot_z3, ave_r8_6keV_reio_z3)
dif_ave_r8_9keV_z3 = frac_diff(ave_r8_9keV_plot_z3, ave_r8_9keV_reio_z3)

#z4 
# first the "mean" one
dif_ave_ave_cdm_z4 = frac_diff(ave_ave_cdm_plot_z4, ave_ave_cdm_reio_z4)
dif_ave_ave_3keV_z4 = frac_diff(ave_ave_3keV_plot_z4, ave_ave_3keV_reio_z4)
dif_ave_ave_4keV_z4 = frac_diff(ave_ave_4keV_plot_z4, ave_ave_4keV_reio_z4)
dif_ave_ave_6keV_z4 = frac_diff(ave_ave_6keV_plot_z4, ave_ave_6keV_reio_z4)
dif_ave_ave_9keV_z4 = frac_diff(ave_ave_9keV_plot_z4, ave_ave_9keV_reio_z4)
# now the realizations, starting with 21cmFAST 
dif_r1_ave_cdm_z4 = frac_diff(r1_ave_cdm_plot_z4, r1_ave_cdm_reio_z4)
dif_r1_ave_3keV_z4 = frac_diff(r1_ave_3keV_plot_z4, r1_ave_3keV_reio_z4)
dif_r1_ave_4keV_z4 = frac_diff(r1_ave_4keV_plot_z4, r1_ave_4keV_reio_z4)
dif_r1_ave_6keV_z4 = frac_diff(r1_ave_6keV_plot_z4, r1_ave_6keV_reio_z4)
dif_r1_ave_9keV_z4 = frac_diff(r1_ave_9keV_plot_z4, r1_ave_9keV_reio_z4)

dif_r2_ave_cdm_z4 = frac_diff(r2_ave_cdm_plot_z4, r2_ave_cdm_reio_z4)
dif_r2_ave_3keV_z4 = frac_diff(r2_ave_3keV_plot_z4, r2_ave_3keV_reio_z4)
dif_r2_ave_4keV_z4 = frac_diff(r2_ave_4keV_plot_z4, r2_ave_4keV_reio_z4)
dif_r2_ave_6keV_z4 = frac_diff(r2_ave_6keV_plot_z4, r2_ave_6keV_reio_z4)
dif_r2_ave_9keV_z4 = frac_diff(r2_ave_9keV_plot_z4, r2_ave_9keV_reio_z4)

dif_r3_ave_cdm_z4 = frac_diff(r3_ave_cdm_plot_z4, r3_ave_cdm_reio_z4)
dif_r3_ave_3keV_z4 = frac_diff(r3_ave_3keV_plot_z4, r3_ave_3keV_reio_z4)
dif_r3_ave_4keV_z4 = frac_diff(r3_ave_4keV_plot_z4, r3_ave_4keV_reio_z4)
dif_r3_ave_6keV_z4 = frac_diff(r3_ave_6keV_plot_z4, r3_ave_6keV_reio_z4)
dif_r3_ave_9keV_z4 = frac_diff(r3_ave_9keV_plot_z4, r3_ave_9keV_reio_z4)

dif_r4_ave_cdm_z4 = frac_diff(r4_ave_cdm_plot_z4, r4_ave_cdm_reio_z4)
dif_r4_ave_3keV_z4 = frac_diff(r4_ave_3keV_plot_z4, r4_ave_3keV_reio_z4)
dif_r4_ave_4keV_z4 = frac_diff(r4_ave_4keV_plot_z4, r4_ave_4keV_reio_z4)
dif_r4_ave_6keV_z4 = frac_diff(r4_ave_6keV_plot_z4, r4_ave_6keV_reio_z4)
dif_r4_ave_9keV_z4 = frac_diff(r4_ave_9keV_plot_z4, r4_ave_9keV_reio_z4)

dif_ave_r1_cdm_z4 = frac_diff(ave_r1_cdm_plot_z4, ave_r1_cdm_reio_z4)
dif_ave_r1_3keV_z4 = frac_diff(ave_r1_3keV_plot_z4, ave_r1_3keV_reio_z4)
dif_ave_r1_4keV_z4 = frac_diff(ave_r1_4keV_plot_z4, ave_r1_4keV_reio_z4)
dif_ave_r1_6keV_z4 = frac_diff(ave_r1_6keV_plot_z4, ave_r1_6keV_reio_z4)
dif_ave_r1_9keV_z4 = frac_diff(ave_r1_9keV_plot_z4, ave_r1_9keV_reio_z4)

dif_ave_r2_cdm_z4 = frac_diff(ave_r2_cdm_plot_z4, ave_r2_cdm_reio_z4)
dif_ave_r2_3keV_z4 = frac_diff(ave_r2_3keV_plot_z4, ave_r2_3keV_reio_z4)
dif_ave_r2_4keV_z4 = frac_diff(ave_r2_4keV_plot_z4, ave_r2_4keV_reio_z4)
dif_ave_r2_6keV_z4 = frac_diff(ave_r2_6keV_plot_z4, ave_r2_6keV_reio_z4)
dif_ave_r2_9keV_z4 = frac_diff(ave_r2_9keV_plot_z4, ave_r2_9keV_reio_z4)

dif_ave_r3_cdm_z4 = frac_diff(ave_r3_cdm_plot_z4, ave_r3_cdm_reio_z4)
dif_ave_r3_3keV_z4 = frac_diff(ave_r3_3keV_plot_z4, ave_r3_3keV_reio_z4)
dif_ave_r3_4keV_z4 = frac_diff(ave_r3_4keV_plot_z4, ave_r3_4keV_reio_z4)
dif_ave_r3_6keV_z4 = frac_diff(ave_r3_6keV_plot_z4, ave_r3_6keV_reio_z4)
dif_ave_r3_9keV_z4 = frac_diff(ave_r3_9keV_plot_z4, ave_r3_9keV_reio_z4)

dif_ave_r4_cdm_z4 = frac_diff(ave_r4_cdm_plot_z4, ave_r4_cdm_reio_z4)
dif_ave_r4_3keV_z4 = frac_diff(ave_r4_3keV_plot_z4, ave_r4_3keV_reio_z4)
dif_ave_r4_4keV_z4 = frac_diff(ave_r4_4keV_plot_z4, ave_r4_4keV_reio_z4)
dif_ave_r4_6keV_z4 = frac_diff(ave_r4_6keV_plot_z4, ave_r4_6keV_reio_z4)
dif_ave_r4_9keV_z4 = frac_diff(ave_r4_9keV_plot_z4, ave_r4_9keV_reio_z4)

dif_ave_r5_cdm_z4 = frac_diff(ave_r5_cdm_plot_z4, ave_r5_cdm_reio_z4)
dif_ave_r5_3keV_z4 = frac_diff(ave_r5_3keV_plot_z4, ave_r5_3keV_reio_z4)
dif_ave_r5_4keV_z4 = frac_diff(ave_r5_4keV_plot_z4, ave_r5_4keV_reio_z4)
dif_ave_r5_6keV_z4 = frac_diff(ave_r5_6keV_plot_z4, ave_r5_6keV_reio_z4)
dif_ave_r5_9keV_z4 = frac_diff(ave_r5_9keV_plot_z4, ave_r5_9keV_reio_z4)

dif_ave_r6_cdm_z4 = frac_diff(ave_r6_cdm_plot_z4, ave_r6_cdm_reio_z4)
dif_ave_r6_3keV_z4 = frac_diff(ave_r6_3keV_plot_z4, ave_r6_3keV_reio_z4)
dif_ave_r6_4keV_z4 = frac_diff(ave_r6_4keV_plot_z4, ave_r6_4keV_reio_z4)
dif_ave_r6_6keV_z4 = frac_diff(ave_r6_6keV_plot_z4, ave_r6_6keV_reio_z4)
dif_ave_r6_9keV_z4 = frac_diff(ave_r6_9keV_plot_z4, ave_r6_9keV_reio_z4)

dif_ave_r7_cdm_z4 = frac_diff(ave_r7_cdm_plot_z4, ave_r7_cdm_reio_z4)
dif_ave_r7_3keV_z4 = frac_diff(ave_r7_3keV_plot_z4, ave_r7_3keV_reio_z4)
dif_ave_r7_4keV_z4 = frac_diff(ave_r7_4keV_plot_z4, ave_r7_4keV_reio_z4)
dif_ave_r7_6keV_z4 = frac_diff(ave_r7_6keV_plot_z4, ave_r7_6keV_reio_z4)
dif_ave_r7_9keV_z4 = frac_diff(ave_r7_9keV_plot_z4, ave_r7_9keV_reio_z4)

dif_ave_r8_cdm_z4 = frac_diff(ave_r8_cdm_plot_z4, ave_r8_cdm_reio_z4)
dif_ave_r8_3keV_z4 = frac_diff(ave_r8_3keV_plot_z4, ave_r8_3keV_reio_z4)
dif_ave_r8_4keV_z4 = frac_diff(ave_r8_4keV_plot_z4, ave_r8_4keV_reio_z4)
dif_ave_r8_6keV_z4 = frac_diff(ave_r8_6keV_plot_z4, ave_r8_6keV_reio_z4)
dif_ave_r8_9keV_z4 = frac_diff(ave_r8_9keV_plot_z4, ave_r8_9keV_reio_z4)

#z5
# first the "mean" one
dif_ave_ave_cdm_z5 = frac_diff(ave_ave_cdm_plot_z5, ave_ave_cdm_reio_z5)
dif_ave_ave_3keV_z5 = frac_diff(ave_ave_3keV_plot_z5, ave_ave_3keV_reio_z5)
dif_ave_ave_4keV_z5 = frac_diff(ave_ave_4keV_plot_z5, ave_ave_4keV_reio_z5)
dif_ave_ave_6keV_z5 = frac_diff(ave_ave_6keV_plot_z5, ave_ave_6keV_reio_z5)
dif_ave_ave_9keV_z5 = frac_diff(ave_ave_9keV_plot_z5, ave_ave_9keV_reio_z5)
# now the realizations, starting with 21cmFAST 
dif_r1_ave_cdm_z5 = frac_diff(r1_ave_cdm_plot_z5, r1_ave_cdm_reio_z5)
dif_r1_ave_3keV_z5 = frac_diff(r1_ave_3keV_plot_z5, r1_ave_3keV_reio_z5)
dif_r1_ave_4keV_z5 = frac_diff(r1_ave_4keV_plot_z5, r1_ave_4keV_reio_z5)
dif_r1_ave_6keV_z5 = frac_diff(r1_ave_6keV_plot_z5, r1_ave_6keV_reio_z5)
dif_r1_ave_9keV_z5 = frac_diff(r1_ave_9keV_plot_z5, r1_ave_9keV_reio_z5)

dif_r2_ave_cdm_z5 = frac_diff(r2_ave_cdm_plot_z5, r2_ave_cdm_reio_z5)
dif_r2_ave_3keV_z5 = frac_diff(r2_ave_3keV_plot_z5, r2_ave_3keV_reio_z5)
dif_r2_ave_4keV_z5 = frac_diff(r2_ave_4keV_plot_z5, r2_ave_4keV_reio_z5)
dif_r2_ave_6keV_z5 = frac_diff(r2_ave_6keV_plot_z5, r2_ave_6keV_reio_z5)
dif_r2_ave_9keV_z5 = frac_diff(r2_ave_9keV_plot_z5, r2_ave_9keV_reio_z5)

dif_r3_ave_cdm_z5 = frac_diff(r3_ave_cdm_plot_z5, r3_ave_cdm_reio_z5)
dif_r3_ave_3keV_z5 = frac_diff(r3_ave_3keV_plot_z5, r3_ave_3keV_reio_z5)
dif_r3_ave_4keV_z5 = frac_diff(r3_ave_4keV_plot_z5, r3_ave_4keV_reio_z5)
dif_r3_ave_6keV_z5 = frac_diff(r3_ave_6keV_plot_z5, r3_ave_6keV_reio_z5)
dif_r3_ave_9keV_z5 = frac_diff(r3_ave_9keV_plot_z5, r3_ave_9keV_reio_z5)

dif_r4_ave_cdm_z5 = frac_diff(r4_ave_cdm_plot_z5, r4_ave_cdm_reio_z5)
dif_r4_ave_3keV_z5 = frac_diff(r4_ave_3keV_plot_z5, r4_ave_3keV_reio_z5)
dif_r4_ave_4keV_z5 = frac_diff(r4_ave_4keV_plot_z5, r4_ave_4keV_reio_z5)
dif_r4_ave_6keV_z5 = frac_diff(r4_ave_6keV_plot_z5, r4_ave_6keV_reio_z5)
dif_r4_ave_9keV_z5 = frac_diff(r4_ave_9keV_plot_z5, r4_ave_9keV_reio_z5)

dif_ave_r1_cdm_z5 = frac_diff(ave_r1_cdm_plot_z5, ave_r1_cdm_reio_z5)
dif_ave_r1_3keV_z5 = frac_diff(ave_r1_3keV_plot_z5, ave_r1_3keV_reio_z5)
dif_ave_r1_4keV_z5 = frac_diff(ave_r1_4keV_plot_z5, ave_r1_4keV_reio_z5)
dif_ave_r1_6keV_z5 = frac_diff(ave_r1_6keV_plot_z5, ave_r1_6keV_reio_z5)
dif_ave_r1_9keV_z5 = frac_diff(ave_r1_9keV_plot_z5, ave_r1_9keV_reio_z5)

dif_ave_r2_cdm_z5 = frac_diff(ave_r2_cdm_plot_z5, ave_r2_cdm_reio_z5)
dif_ave_r2_3keV_z5 = frac_diff(ave_r2_3keV_plot_z5, ave_r2_3keV_reio_z5)
dif_ave_r2_4keV_z5 = frac_diff(ave_r2_4keV_plot_z5, ave_r2_4keV_reio_z5)
dif_ave_r2_6keV_z5 = frac_diff(ave_r2_6keV_plot_z5, ave_r2_6keV_reio_z5)
dif_ave_r2_9keV_z5 = frac_diff(ave_r2_9keV_plot_z5, ave_r2_9keV_reio_z5)

dif_ave_r3_cdm_z5 = frac_diff(ave_r3_cdm_plot_z5, ave_r3_cdm_reio_z5)
dif_ave_r3_3keV_z5 = frac_diff(ave_r3_3keV_plot_z5, ave_r3_3keV_reio_z5)
dif_ave_r3_4keV_z5 = frac_diff(ave_r3_4keV_plot_z5, ave_r3_4keV_reio_z5)
dif_ave_r3_6keV_z5 = frac_diff(ave_r3_6keV_plot_z5, ave_r3_6keV_reio_z5)
dif_ave_r3_9keV_z5 = frac_diff(ave_r3_9keV_plot_z5, ave_r3_9keV_reio_z5)

dif_ave_r4_cdm_z5 = frac_diff(ave_r4_cdm_plot_z5, ave_r4_cdm_reio_z5)
dif_ave_r4_3keV_z5 = frac_diff(ave_r4_3keV_plot_z5, ave_r4_3keV_reio_z5)
dif_ave_r4_4keV_z5 = frac_diff(ave_r4_4keV_plot_z5, ave_r4_4keV_reio_z5)
dif_ave_r4_6keV_z5 = frac_diff(ave_r4_6keV_plot_z5, ave_r4_6keV_reio_z5)
dif_ave_r4_9keV_z5 = frac_diff(ave_r4_9keV_plot_z5, ave_r4_9keV_reio_z5)

dif_ave_r5_cdm_z5 = frac_diff(ave_r5_cdm_plot_z5, ave_r5_cdm_reio_z5)
dif_ave_r5_3keV_z5 = frac_diff(ave_r5_3keV_plot_z5, ave_r5_3keV_reio_z5)
dif_ave_r5_4keV_z5 = frac_diff(ave_r5_4keV_plot_z5, ave_r5_4keV_reio_z5)
dif_ave_r5_6keV_z5 = frac_diff(ave_r5_6keV_plot_z5, ave_r5_6keV_reio_z5)
dif_ave_r5_9keV_z5 = frac_diff(ave_r5_9keV_plot_z5, ave_r5_9keV_reio_z5)

dif_ave_r6_cdm_z5 = frac_diff(ave_r6_cdm_plot_z5, ave_r6_cdm_reio_z5)
dif_ave_r6_3keV_z5 = frac_diff(ave_r6_3keV_plot_z5, ave_r6_3keV_reio_z5)
dif_ave_r6_4keV_z5 = frac_diff(ave_r6_4keV_plot_z5, ave_r6_4keV_reio_z5)
dif_ave_r6_6keV_z5 = frac_diff(ave_r6_6keV_plot_z5, ave_r6_6keV_reio_z5)
dif_ave_r6_9keV_z5 = frac_diff(ave_r6_9keV_plot_z5, ave_r6_9keV_reio_z5)

dif_ave_r7_cdm_z5 = frac_diff(ave_r7_cdm_plot_z5, ave_r7_cdm_reio_z5)
dif_ave_r7_3keV_z5 = frac_diff(ave_r7_3keV_plot_z5, ave_r7_3keV_reio_z5)
dif_ave_r7_4keV_z5 = frac_diff(ave_r7_4keV_plot_z5, ave_r7_4keV_reio_z5)
dif_ave_r7_6keV_z5 = frac_diff(ave_r7_6keV_plot_z5, ave_r7_6keV_reio_z5)
dif_ave_r7_9keV_z5 = frac_diff(ave_r7_9keV_plot_z5, ave_r7_9keV_reio_z5)

dif_ave_r8_cdm_z5 = frac_diff(ave_r8_cdm_plot_z5, ave_r8_cdm_reio_z5)
dif_ave_r8_3keV_z5 = frac_diff(ave_r8_3keV_plot_z5, ave_r8_3keV_reio_z5)
dif_ave_r8_4keV_z5 = frac_diff(ave_r8_4keV_plot_z5, ave_r8_4keV_reio_z5)
dif_ave_r8_6keV_z5 = frac_diff(ave_r8_6keV_plot_z5, ave_r8_6keV_reio_z5)
dif_ave_r8_9keV_z5 = frac_diff(ave_r8_9keV_plot_z5, ave_r8_9keV_reio_z5)

# now we can compute the sample variance on the fractional difference
dif_ave_ave_cdm_error_z1 = sample_variance(dif_ave_ave_cdm_z1, dif_r1_ave_cdm_z1, dif_r2_ave_cdm_z1, dif_r3_ave_cdm_z1, dif_r4_ave_cdm_z1, dif_ave_r1_cdm_z1, dif_ave_r2_cdm_z1, dif_ave_r3_cdm_z1, dif_ave_r4_cdm_z1,dif_ave_r5_cdm_z1, dif_ave_r6_cdm_z1, dif_ave_r7_cdm_z1, dif_ave_r8_cdm_z1)
dif_ave_ave_3keV_error_z1 = sample_variance(dif_ave_ave_3keV_z1, dif_r1_ave_3keV_z1, dif_r2_ave_3keV_z1, dif_r3_ave_3keV_z1, dif_r4_ave_3keV_z1, dif_ave_r1_3keV_z1, dif_ave_r2_3keV_z1, dif_ave_r3_3keV_z1, dif_ave_r4_3keV_z1,dif_ave_r5_3keV_z1, dif_ave_r6_3keV_z1, dif_ave_r7_3keV_z1, dif_ave_r8_3keV_z1)
dif_ave_ave_4keV_error_z1 = sample_variance(dif_ave_ave_4keV_z1, dif_r1_ave_4keV_z1, dif_r2_ave_4keV_z1, dif_r3_ave_4keV_z1, dif_r4_ave_4keV_z1, dif_ave_r1_4keV_z1, dif_ave_r2_4keV_z1, dif_ave_r3_4keV_z1, dif_ave_r4_4keV_z1,dif_ave_r5_4keV_z1, dif_ave_r6_4keV_z1, dif_ave_r7_4keV_z1, dif_ave_r8_4keV_z1)
dif_ave_ave_6keV_error_z1 = sample_variance(dif_ave_ave_6keV_z1, dif_r1_ave_6keV_z1, dif_r2_ave_6keV_z1, dif_r3_ave_6keV_z1, dif_r4_ave_6keV_z1, dif_ave_r1_6keV_z1, dif_ave_r2_6keV_z1, dif_ave_r3_6keV_z1, dif_ave_r4_6keV_z1,dif_ave_r5_6keV_z1, dif_ave_r6_6keV_z1, dif_ave_r7_6keV_z1, dif_ave_r8_6keV_z1)
dif_ave_ave_9keV_error_z1 = sample_variance(dif_ave_ave_9keV_z1, dif_r1_ave_9keV_z1, dif_r2_ave_9keV_z1, dif_r3_ave_9keV_z1, dif_r4_ave_9keV_z1, dif_ave_r1_9keV_z1, dif_ave_r2_9keV_z1, dif_ave_r3_9keV_z1, dif_ave_r4_9keV_z1,dif_ave_r5_9keV_z1, dif_ave_r6_9keV_z1, dif_ave_r7_9keV_z1, dif_ave_r8_9keV_z1)
# z2
dif_ave_ave_cdm_error_z2 = sample_variance(dif_ave_ave_cdm_z2, dif_r1_ave_cdm_z2, dif_r2_ave_cdm_z2, dif_r3_ave_cdm_z2, dif_r4_ave_cdm_z2, dif_ave_r1_cdm_z2, dif_ave_r2_cdm_z2, dif_ave_r3_cdm_z2, dif_ave_r4_cdm_z2,dif_ave_r5_cdm_z2, dif_ave_r6_cdm_z2, dif_ave_r7_cdm_z2, dif_ave_r8_cdm_z2)
dif_ave_ave_3keV_error_z2 = sample_variance(dif_ave_ave_3keV_z2, dif_r1_ave_3keV_z2, dif_r2_ave_3keV_z2, dif_r3_ave_3keV_z2, dif_r4_ave_3keV_z2, dif_ave_r1_3keV_z2, dif_ave_r2_3keV_z2, dif_ave_r3_3keV_z2, dif_ave_r4_3keV_z2,dif_ave_r5_3keV_z2, dif_ave_r6_3keV_z2, dif_ave_r7_3keV_z2, dif_ave_r8_3keV_z2)
dif_ave_ave_4keV_error_z2 = sample_variance(dif_ave_ave_4keV_z2, dif_r1_ave_4keV_z2, dif_r2_ave_4keV_z2, dif_r3_ave_4keV_z2, dif_r4_ave_4keV_z2, dif_ave_r1_4keV_z2, dif_ave_r2_4keV_z2, dif_ave_r3_4keV_z2, dif_ave_r4_4keV_z2,dif_ave_r5_4keV_z2, dif_ave_r6_4keV_z2, dif_ave_r7_4keV_z2, dif_ave_r8_4keV_z2)
dif_ave_ave_6keV_error_z2 = sample_variance(dif_ave_ave_6keV_z2, dif_r1_ave_6keV_z2, dif_r2_ave_6keV_z2, dif_r3_ave_6keV_z2, dif_r4_ave_6keV_z2, dif_ave_r1_6keV_z2, dif_ave_r2_6keV_z2, dif_ave_r3_6keV_z2, dif_ave_r4_6keV_z2,dif_ave_r5_6keV_z2, dif_ave_r6_6keV_z2, dif_ave_r7_6keV_z2, dif_ave_r8_6keV_z2)
dif_ave_ave_9keV_error_z2 = sample_variance(dif_ave_ave_9keV_z2, dif_r1_ave_9keV_z2, dif_r2_ave_9keV_z2, dif_r3_ave_9keV_z2, dif_r4_ave_9keV_z2, dif_ave_r1_9keV_z2, dif_ave_r2_9keV_z2, dif_ave_r3_9keV_z2, dif_ave_r4_9keV_z2,dif_ave_r5_9keV_z2, dif_ave_r6_9keV_z2, dif_ave_r7_9keV_z2, dif_ave_r8_9keV_z2)
# z3
dif_ave_ave_cdm_error_z3 = sample_variance(dif_ave_ave_cdm_z3, dif_r1_ave_cdm_z3, dif_r2_ave_cdm_z3, dif_r3_ave_cdm_z3, dif_r4_ave_cdm_z3, dif_ave_r1_cdm_z3, dif_ave_r2_cdm_z3, dif_ave_r3_cdm_z3, dif_ave_r4_cdm_z3,dif_ave_r5_cdm_z3, dif_ave_r6_cdm_z3, dif_ave_r7_cdm_z3, dif_ave_r8_cdm_z3)
dif_ave_ave_3keV_error_z3 = sample_variance(dif_ave_ave_3keV_z3, dif_r1_ave_3keV_z3, dif_r2_ave_3keV_z3, dif_r3_ave_3keV_z3, dif_r4_ave_3keV_z3, dif_ave_r1_3keV_z3, dif_ave_r2_3keV_z3, dif_ave_r3_3keV_z3, dif_ave_r4_3keV_z3,dif_ave_r5_3keV_z3, dif_ave_r6_3keV_z3, dif_ave_r7_3keV_z3, dif_ave_r8_3keV_z3)
dif_ave_ave_4keV_error_z3 = sample_variance(dif_ave_ave_4keV_z3, dif_r1_ave_4keV_z3, dif_r2_ave_4keV_z3, dif_r3_ave_4keV_z3, dif_r4_ave_4keV_z3, dif_ave_r1_4keV_z3, dif_ave_r2_4keV_z3, dif_ave_r3_4keV_z3, dif_ave_r4_4keV_z3,dif_ave_r5_4keV_z3, dif_ave_r6_4keV_z3, dif_ave_r7_4keV_z3, dif_ave_r8_4keV_z3)
dif_ave_ave_6keV_error_z3 = sample_variance(dif_ave_ave_6keV_z3, dif_r1_ave_6keV_z3, dif_r2_ave_6keV_z3, dif_r3_ave_6keV_z3, dif_r4_ave_6keV_z3, dif_ave_r1_6keV_z3, dif_ave_r2_6keV_z3, dif_ave_r3_6keV_z3, dif_ave_r4_6keV_z3,dif_ave_r5_6keV_z3, dif_ave_r6_6keV_z3, dif_ave_r7_6keV_z3, dif_ave_r8_6keV_z3)
dif_ave_ave_9keV_error_z3 = sample_variance(dif_ave_ave_9keV_z3, dif_r1_ave_9keV_z3, dif_r2_ave_9keV_z3, dif_r3_ave_9keV_z3, dif_r4_ave_9keV_z3, dif_ave_r1_9keV_z3, dif_ave_r2_9keV_z3, dif_ave_r3_9keV_z3, dif_ave_r4_9keV_z3,dif_ave_r5_9keV_z3, dif_ave_r6_9keV_z3, dif_ave_r7_9keV_z3, dif_ave_r8_9keV_z3)
# z4
dif_ave_ave_cdm_error_z4 = sample_variance(dif_ave_ave_cdm_z4, dif_r1_ave_cdm_z4, dif_r2_ave_cdm_z4, dif_r3_ave_cdm_z4, dif_r4_ave_cdm_z4, dif_ave_r1_cdm_z4, dif_ave_r2_cdm_z4, dif_ave_r3_cdm_z4, dif_ave_r4_cdm_z4,dif_ave_r5_cdm_z4, dif_ave_r6_cdm_z4, dif_ave_r7_cdm_z4, dif_ave_r8_cdm_z4)
dif_ave_ave_3keV_error_z4 = sample_variance(dif_ave_ave_3keV_z4, dif_r1_ave_3keV_z4, dif_r2_ave_3keV_z4, dif_r3_ave_3keV_z4, dif_r4_ave_3keV_z4, dif_ave_r1_3keV_z4, dif_ave_r2_3keV_z4, dif_ave_r3_3keV_z4, dif_ave_r4_3keV_z4,dif_ave_r5_3keV_z4, dif_ave_r6_3keV_z4, dif_ave_r7_3keV_z4, dif_ave_r8_3keV_z4)
dif_ave_ave_4keV_error_z4 = sample_variance(dif_ave_ave_4keV_z4, dif_r1_ave_4keV_z4, dif_r2_ave_4keV_z4, dif_r3_ave_4keV_z4, dif_r4_ave_4keV_z4, dif_ave_r1_4keV_z4, dif_ave_r2_4keV_z4, dif_ave_r3_4keV_z4, dif_ave_r4_4keV_z4,dif_ave_r5_4keV_z4, dif_ave_r6_4keV_z4, dif_ave_r7_4keV_z4, dif_ave_r8_4keV_z4)
dif_ave_ave_6keV_error_z4 = sample_variance(dif_ave_ave_6keV_z4, dif_r1_ave_6keV_z4, dif_r2_ave_6keV_z4, dif_r3_ave_6keV_z4, dif_r4_ave_6keV_z4, dif_ave_r1_6keV_z4, dif_ave_r2_6keV_z4, dif_ave_r3_6keV_z4, dif_ave_r4_6keV_z4,dif_ave_r5_6keV_z4, dif_ave_r6_6keV_z4, dif_ave_r7_6keV_z4, dif_ave_r8_6keV_z4)
dif_ave_ave_9keV_error_z4 = sample_variance(dif_ave_ave_9keV_z4, dif_r1_ave_9keV_z4, dif_r2_ave_9keV_z4, dif_r3_ave_9keV_z4, dif_r4_ave_9keV_z4, dif_ave_r1_9keV_z4, dif_ave_r2_9keV_z4, dif_ave_r3_9keV_z4, dif_ave_r4_9keV_z4,dif_ave_r5_9keV_z4, dif_ave_r6_9keV_z4, dif_ave_r7_9keV_z4, dif_ave_r8_9keV_z4)
# z5
dif_ave_ave_cdm_error_z5 = sample_variance(dif_ave_ave_cdm_z5, dif_r1_ave_cdm_z5, dif_r2_ave_cdm_z5, dif_r3_ave_cdm_z5, dif_r4_ave_cdm_z5, dif_ave_r1_cdm_z5, dif_ave_r2_cdm_z5, dif_ave_r3_cdm_z5, dif_ave_r4_cdm_z5,dif_ave_r5_cdm_z5, dif_ave_r6_cdm_z5, dif_ave_r7_cdm_z5, dif_ave_r8_cdm_z5)
dif_ave_ave_3keV_error_z5 = sample_variance(dif_ave_ave_3keV_z5, dif_r1_ave_3keV_z5, dif_r2_ave_3keV_z5, dif_r3_ave_3keV_z5, dif_r4_ave_3keV_z5, dif_ave_r1_3keV_z5, dif_ave_r2_3keV_z5, dif_ave_r3_3keV_z5, dif_ave_r4_3keV_z5,dif_ave_r5_3keV_z5, dif_ave_r6_3keV_z5, dif_ave_r7_3keV_z5, dif_ave_r8_3keV_z5)
dif_ave_ave_4keV_error_z5 = sample_variance(dif_ave_ave_4keV_z5, dif_r1_ave_4keV_z5, dif_r2_ave_4keV_z5, dif_r3_ave_4keV_z5, dif_r4_ave_4keV_z5, dif_ave_r1_4keV_z5, dif_ave_r2_4keV_z5, dif_ave_r3_4keV_z5, dif_ave_r4_4keV_z5,dif_ave_r5_4keV_z5, dif_ave_r6_4keV_z5, dif_ave_r7_4keV_z5, dif_ave_r8_4keV_z5)
dif_ave_ave_6keV_error_z5 = sample_variance(dif_ave_ave_6keV_z5, dif_r1_ave_6keV_z5, dif_r2_ave_6keV_z5, dif_r3_ave_6keV_z5, dif_r4_ave_6keV_z5, dif_ave_r1_6keV_z5, dif_ave_r2_6keV_z5, dif_ave_r3_6keV_z5, dif_ave_r4_6keV_z5,dif_ave_r5_6keV_z5, dif_ave_r6_6keV_z5, dif_ave_r7_6keV_z5, dif_ave_r8_6keV_z5)
dif_ave_ave_9keV_error_z5 = sample_variance(dif_ave_ave_9keV_z5, dif_r1_ave_9keV_z5, dif_r2_ave_9keV_z5, dif_r3_ave_9keV_z5, dif_r4_ave_9keV_z5, dif_ave_r1_9keV_z5, dif_ave_r2_9keV_z5, dif_ave_r3_9keV_z5, dif_ave_r4_9keV_z5,dif_ave_r5_9keV_z5, dif_ave_r6_9keV_z5, dif_ave_r7_9keV_z5, dif_ave_r8_9keV_z5)
n = 5
colors = plt.cm.viridis(np.linspace(0,1,n))
labels = [r'CDM', r'm$_{\rm wdm} = 9$ keV', r'm$_{\rm wdm} = 6$ keV', r'm$_{\rm wdm} = 4$ keV', r'm$_{\rm wdm} = 3$ keV']
mean_table_z1 = np.zeros((len(k_array),n))
error_table_z1 = np.zeros((len(k_array),n))
mean_table_z2 = np.zeros((len(k_array),n))
error_table_z2 = np.zeros((len(k_array),n))
mean_table_z3 = np.zeros((len(k_array),n))
error_table_z3 = np.zeros((len(k_array),n))
mean_table_z4 = np.zeros((len(k_array),n))
error_table_z4 = np.zeros((len(k_array),n))
mean_table_z5 = np.zeros((len(k_array),n))
error_table_z5 = np.zeros((len(k_array),n))

matter_table_z1 = np.zeros((len(k_array),n))
matter_table_z2 = np.zeros((len(k_array),n))
matter_table_z3 = np.zeros((len(k_array),n))
matter_table_z4 = np.zeros((len(k_array),n))
matter_table_z5 = np.zeros((len(k_array),n))

matter_table_z1[:,0] = np.array([ave_ave_cdm.FluxP3D_lya_Mpc(z1, k, 0.1) for k in k_array])
matter_table_z1[:,1] = np.array([ave_ave_3keV.FluxP3D_lya_Mpc(z1, k, 0.1) for k in k_array])
matter_table_z1[:,2] = np.array([ave_ave_4keV.FluxP3D_lya_Mpc(z1, k, 0.1) for k in k_array])
matter_table_z1[:,3] = np.array([ave_ave_6keV.FluxP3D_lya_Mpc(z1, k, 0.1) for k in k_array])
matter_table_z1[:,4] = np.array([ave_ave_9keV.FluxP3D_lya_Mpc(z1, k, 0.1) for k in k_array])

matter_table_z2[:,0] = np.array([ave_ave_cdm.FluxP3D_lya_Mpc(z2, k, 0.1) for k in k_array])
matter_table_z2[:,1] = np.array([ave_ave_3keV.FluxP3D_lya_Mpc(z2, k, 0.1) for k in k_array])
matter_table_z2[:,2] = np.array([ave_ave_4keV.FluxP3D_lya_Mpc(z2, k, 0.1) for k in k_array])
matter_table_z2[:,3] = np.array([ave_ave_6keV.FluxP3D_lya_Mpc(z2, k, 0.1) for k in k_array])
matter_table_z2[:,4] = np.array([ave_ave_9keV.FluxP3D_lya_Mpc(z2, k, 0.1) for k in k_array])

matter_table_z3[:,0] = np.array([ave_ave_cdm.FluxP3D_lya_Mpc(z3, k, 0.1) for k in k_array])
matter_table_z3[:,1] = np.array([ave_ave_3keV.FluxP3D_lya_Mpc(z3, k, 0.1) for k in k_array])
matter_table_z3[:,2] = np.array([ave_ave_4keV.FluxP3D_lya_Mpc(z3, k, 0.1) for k in k_array])
matter_table_z3[:,3] = np.array([ave_ave_6keV.FluxP3D_lya_Mpc(z3, k, 0.1) for k in k_array])
matter_table_z3[:,4] = np.array([ave_ave_9keV.FluxP3D_lya_Mpc(z3, k, 0.1) for k in k_array])

matter_table_z4[:,0] = np.array([ave_ave_cdm.FluxP3D_lya_Mpc(z4, k, 0.1) for k in k_array])
matter_table_z4[:,1] = np.array([ave_ave_3keV.FluxP3D_lya_Mpc(z4, k, 0.1) for k in k_array])
matter_table_z4[:,2] = np.array([ave_ave_4keV.FluxP3D_lya_Mpc(z4, k, 0.1) for k in k_array])
matter_table_z4[:,3] = np.array([ave_ave_6keV.FluxP3D_lya_Mpc(z4, k, 0.1) for k in k_array])
matter_table_z4[:,4] = np.array([ave_ave_9keV.FluxP3D_lya_Mpc(z4, k, 0.1) for k in k_array])

matter_table_z5[:,0] = np.array([ave_ave_cdm.FluxP3D_lya_Mpc(z5, k, 0.1) for k in k_array])
matter_table_z5[:,1] = np.array([ave_ave_3keV.FluxP3D_lya_Mpc(z5, k, 0.1) for k in k_array])
matter_table_z5[:,2] = np.array([ave_ave_4keV.FluxP3D_lya_Mpc(z5, k, 0.1) for k in k_array])
matter_table_z5[:,3] = np.array([ave_ave_6keV.FluxP3D_lya_Mpc(z5, k, 0.1) for k in k_array])
matter_table_z5[:,4] = np.array([ave_ave_9keV.FluxP3D_lya_Mpc(z5, k, 0.1) for k in k_array])

mean_table_z1[:,0] = ave_ave_cdm_plot_z1[:]
mean_table_z1[:,1] = ave_ave_9keV_plot_z1[:]
mean_table_z1[:,2] = ave_ave_6keV_plot_z1[:]
mean_table_z1[:,3] = ave_ave_4keV_plot_z1[:]
mean_table_z1[:,4] = ave_ave_3keV_plot_z1[:]

error_table_z1[:,0] = ave_ave_cdm_error_z1[:]
error_table_z1[:,1] = ave_ave_9keV_error_z1[:]
error_table_z1[:,2] = ave_ave_6keV_error_z1[:]
error_table_z1[:,3] = ave_ave_4keV_error_z1[:]
error_table_z1[:,4] = ave_ave_3keV_error_z1[:]

mean_table_z2[:,0] = ave_ave_cdm_plot_z2[:]
mean_table_z2[:,1] = ave_ave_9keV_plot_z2[:]
mean_table_z2[:,2] = ave_ave_6keV_plot_z2[:]
mean_table_z2[:,3] = ave_ave_4keV_plot_z2[:]
mean_table_z2[:,4] = ave_ave_3keV_plot_z2[:]

error_table_z2[:,0] = ave_ave_cdm_error_z2[:]
error_table_z2[:,1] = ave_ave_9keV_error_z2[:]
error_table_z2[:,2] = ave_ave_6keV_error_z2[:]
error_table_z2[:,3] = ave_ave_4keV_error_z2[:]
error_table_z2[:,4] = ave_ave_3keV_error_z2[:]

mean_table_z3[:,0] = ave_ave_cdm_plot_z3[:]
mean_table_z3[:,1] = ave_ave_9keV_plot_z3[:]
mean_table_z3[:,2] = ave_ave_6keV_plot_z3[:]
mean_table_z3[:,3] = ave_ave_4keV_plot_z3[:]
mean_table_z3[:,4] = ave_ave_3keV_plot_z3[:]

error_table_z3[:,0] = ave_ave_cdm_error_z3[:]
error_table_z3[:,1] = ave_ave_9keV_error_z3[:]
error_table_z3[:,2] = ave_ave_6keV_error_z3[:]
error_table_z3[:,3] = ave_ave_4keV_error_z3[:]
error_table_z3[:,4] = ave_ave_3keV_error_z3[:]

mean_table_z4[:,0] = ave_ave_cdm_plot_z4[:]
mean_table_z4[:,1] = ave_ave_9keV_plot_z4[:]
mean_table_z4[:,2] = ave_ave_6keV_plot_z4[:]
mean_table_z4[:,3] = ave_ave_4keV_plot_z4[:]
mean_table_z4[:,4] = ave_ave_3keV_plot_z4[:]

error_table_z4[:,0] = ave_ave_cdm_error_z4[:]
error_table_z4[:,1] = ave_ave_9keV_error_z4[:]
error_table_z4[:,2] = ave_ave_6keV_error_z4[:]
error_table_z4[:,3] = ave_ave_4keV_error_z4[:]
error_table_z4[:,4] = ave_ave_3keV_error_z4[:]

mean_table_z5[:,0] = ave_ave_cdm_plot_z5[:]
mean_table_z5[:,1] = ave_ave_9keV_plot_z5[:]
mean_table_z5[:,2] = ave_ave_6keV_plot_z5[:]
mean_table_z5[:,3] = ave_ave_4keV_plot_z5[:]
mean_table_z5[:,4] = ave_ave_3keV_plot_z5[:]

error_table_z5[:,0] = ave_ave_cdm_error_z5[:]
error_table_z5[:,1] = ave_ave_9keV_error_z5[:]
error_table_z5[:,2] = ave_ave_6keV_error_z5[:]
error_table_z5[:,3] = ave_ave_4keV_error_z5[:]
error_table_z5[:,4] = ave_ave_3keV_error_z5[:]


dif_colors = plt.cm.viridis(np.linspace(0,1,n))
dif_labels = [r'CDM', r'm$_{\rm wdm} = 9$ keV', r'm$_{\rm wdm} = 6$ keV', r'm$_{\rm wdm} = 4$ keV', r'm$_{\rm wdm} = 3$ keV']
dif_mean_z1 = np.zeros((len(k_array),n))
dif_error_z1 = np.zeros((len(k_array),n))
dif_mean_z2 = np.zeros((len(k_array),n))
dif_error_z2 = np.zeros((len(k_array),n))
dif_mean_z3 = np.zeros((len(k_array),n))
dif_error_z3 = np.zeros((len(k_array),n))
dif_mean_z4 = np.zeros((len(k_array),n))
dif_error_z4 = np.zeros((len(k_array),n))
dif_mean_z5 = np.zeros((len(k_array),n))
dif_error_z5 = np.zeros((len(k_array),n))

dif_mean_z1[:,0] = dif_ave_ave_cdm_z1[:]
dif_mean_z1[:,1] = dif_ave_ave_9keV_z1[:]
dif_mean_z1[:,2] = dif_ave_ave_6keV_z1[:]
dif_mean_z1[:,3] = dif_ave_ave_4keV_z1[:]
dif_mean_z1[:,4] = dif_ave_ave_3keV_z1[:]

dif_error_z1[:,0] = dif_ave_ave_cdm_error_z1[:]
dif_error_z1[:,1] = dif_ave_ave_9keV_error_z1[:]
dif_error_z1[:,2] = dif_ave_ave_6keV_error_z1[:]
dif_error_z1[:,3] = dif_ave_ave_4keV_error_z1[:]
dif_error_z1[:,4] = dif_ave_ave_3keV_error_z1[:]

dif_mean_z2[:,0] = dif_ave_ave_cdm_z2[:]
dif_mean_z2[:,1] = dif_ave_ave_9keV_z2[:]
dif_mean_z2[:,2] = dif_ave_ave_6keV_z2[:]
dif_mean_z2[:,3] = dif_ave_ave_4keV_z2[:]
dif_mean_z2[:,4] = dif_ave_ave_3keV_z2[:]

dif_error_z2[:,0] = dif_ave_ave_cdm_error_z2[:]
dif_error_z2[:,1] = dif_ave_ave_9keV_error_z2[:]
dif_error_z2[:,2] = dif_ave_ave_6keV_error_z2[:]
dif_error_z2[:,3] = dif_ave_ave_4keV_error_z2[:]
dif_error_z2[:,4] = dif_ave_ave_3keV_error_z2[:]

dif_mean_z3[:,0] = dif_ave_ave_cdm_z3[:]
dif_mean_z3[:,1] = dif_ave_ave_9keV_z3[:]
dif_mean_z3[:,2] = dif_ave_ave_6keV_z3[:]
dif_mean_z3[:,3] = dif_ave_ave_4keV_z3[:]
dif_mean_z3[:,4] = dif_ave_ave_3keV_z3[:]

dif_error_z3[:,0] = dif_ave_ave_cdm_error_z3[:]
dif_error_z3[:,1] = dif_ave_ave_9keV_error_z3[:]
dif_error_z3[:,2] = dif_ave_ave_6keV_error_z3[:]
dif_error_z3[:,3] = dif_ave_ave_4keV_error_z3[:]
dif_error_z3[:,4] = dif_ave_ave_3keV_error_z3[:]

dif_mean_z4[:,0] = dif_ave_ave_cdm_z4[:]
dif_mean_z4[:,1] = dif_ave_ave_9keV_z4[:]
dif_mean_z4[:,2] = dif_ave_ave_6keV_z4[:]
dif_mean_z4[:,3] = dif_ave_ave_4keV_z4[:]
dif_mean_z4[:,4] = dif_ave_ave_3keV_z4[:]

dif_error_z4[:,0] = dif_ave_ave_cdm_error_z4[:]
dif_error_z4[:,1] = dif_ave_ave_9keV_error_z4[:]
dif_error_z4[:,2] = dif_ave_ave_6keV_error_z4[:]
dif_error_z4[:,3] = dif_ave_ave_4keV_error_z4[:]
dif_error_z4[:,4] = dif_ave_ave_3keV_error_z4[:]

dif_mean_z5[:,0] = dif_ave_ave_cdm_z5[:]
dif_mean_z5[:,1] = dif_ave_ave_9keV_z5[:]
dif_mean_z5[:,2] = dif_ave_ave_6keV_z5[:]
dif_mean_z5[:,3] = dif_ave_ave_4keV_z5[:]
dif_mean_z5[:,4] = dif_ave_ave_3keV_z5[:]

dif_error_z5[:,0] = dif_ave_ave_cdm_error_z5[:]
dif_error_z5[:,1] = dif_ave_ave_9keV_error_z5[:]
dif_error_z5[:,2] = dif_ave_ave_6keV_error_z5[:]
dif_error_z5[:,3] = dif_ave_ave_4keV_error_z5[:]
dif_error_z5[:,4] = dif_ave_ave_3keV_error_z5[:]

# plotting, need two columns one per redshift
fig, axs = plt.subplots(2, 5,figsize=(25,10), sharex=True)
for i in range(n):
    axs[0,0].fill_between(k_array, mean_table_z1[:,i] - error_table_z1[:,i], mean_table_z1[:,i] + error_table_z1[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[0,1].fill_between(k_array, mean_table_z2[:,i] - error_table_z2[:,i], mean_table_z2[:,i] + error_table_z2[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[0,2].fill_between(k_array, mean_table_z3[:,i] - error_table_z3[:,i], mean_table_z3[:,i] + error_table_z3[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[0,3].fill_between(k_array, mean_table_z4[:,i] - error_table_z4[:,i], mean_table_z4[:,i] + error_table_z4[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[0,4].fill_between(k_array, mean_table_z5[:,i] - error_table_z5[:,i], mean_table_z5[:,i] + error_table_z5[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[0,0].plot(k_array, matter_table_z1[:,i], '--', color = colors[i])
    axs[0,1].plot(k_array, matter_table_z2[:,i], '--', color = colors[i])
    axs[0,2].plot(k_array, matter_table_z3[:,i], '--', color = colors[i])
    axs[0,3].plot(k_array, matter_table_z4[:,i], '--', color = colors[i])
    axs[0,4].plot(k_array, matter_table_z5[:,i], '--', color = colors[i])

# axs[0,0].errorbar(k_array, ave_ave_cdm_plot_z1, yerr=ave_ave_cdm_error_z1, fmt='-o', color='royalblue', label=r'CDM, $z = 2$')
# axs[0,0].errorbar(k_array, ave_ave_3keV_plot_z1, yerr=ave_ave_3keV_error_z1, fmt=':s', color='magenta', label=r'$m_{\rm wdm} = 3$ keV, $z = 2$')
# axs[0,0].errorbar(k_array, ave_ave_4keV_plot_z1, yerr=ave_ave_4keV_error_z1, fmt=':d', color='orange', label=r'$m_{\rm wdm} = 4$ keV, $z = 2$')
# axs[0,0].errorbar(k_array, ave_ave_6keV_plot_z1, yerr=ave_ave_6keV_error_z1, fmt='--p', color='purple', label=r'$m_{\rm wdm} = 6$ keV, $z = 2$')
# axs[0,0].errorbar(k_array, ave_ave_9keV_plot_z1, yerr=ave_ave_9keV_error_z1, fmt='-.*', color='green', label=r'$m_{\rm wdm} = 9$ keV, $z = 2$')
axs[0,0].grid(linestyle='dotted')
axs[0,0].set_title('Redshift = 2.0', fontsize=16)
axs[0,0].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[0,0].set_ylabel(r'3D Flux power spectrum    [Mpc]', fontsize=16)
axs[0,0].set_xscale('log')
axs[0,0].legend(loc='best',fontsize=14)
axs[0,0].set_xlim(0.05, 1.1)
#axs[0,0].set_ylim(0,100)
# other redshift
# axs[0,1].errorbar(k_array, ave_ave_cdm_plot_z5, yerr=ave_ave_cdm_error_z5, fmt='-o', color='royalblue', label=r'CDM, $z = 4$')
# axs[0,1].errorbar(k_array, ave_ave_3keV_plot_z5, yerr=ave_ave_3keV_error_z5, fmt=':s', color='magenta', label=r'$m_{\rm wdm} = 3$ keV, $z = 4$')
# axs[0,1].errorbar(k_array, ave_ave_4keV_plot_z5, yerr=ave_ave_4keV_error_z5, fmt=':d', color='orange', label=r'$m_{\rm wdm} = 4$ keV, $z = 4$')
# axs[0,1].errorbar(k_array, ave_ave_6keV_plot_z5, yerr=ave_ave_6keV_error_z5, fmt='--p', color='purple', label=r'$m_{\rm wdm} = 6$ keV, $z = 4$')
# axs[0,1].errorbar(k_array, ave_ave_9keV_plot_z5, yerr=ave_ave_9keV_error_z5, fmt='-.*', color='green', label=r'$m_{\rm wdm} = 9$ keV, $z = 4$')
axs[0,1].grid(linestyle='dotted')
axs[0,1].set_title('Redshift = 2.5', fontsize=16)
axs[0,1].set_xscale('log')
axs[0,1].tick_params(axis = 'both', which = 'major', labelsize = 14)
#axs[0,1].set_ylim(0,1000)
axs[0,2].grid(linestyle='dotted')
axs[0,2].set_title('Redshift = 3.0', fontsize=16)
axs[0,2].set_xscale('log')
axs[0,2].tick_params(axis = 'both', which = 'major', labelsize = 14)
#axs[0,2].set_ylim(0,1000)
axs[0,3].grid(linestyle='dotted')
axs[0,3].set_title('Redshift = 3.5', fontsize=16)
axs[0,3].set_xscale('log')
axs[0,3].tick_params(axis = 'both', which = 'major', labelsize = 14)
#axs[0,3].set_ylim(0,1000)
axs[0,4].grid(linestyle='dotted')
axs[0,4].set_title('Redshift = 4.0', fontsize=16)
axs[0,4].set_xscale('log')
axs[0,4].tick_params(axis = 'both', which = 'major', labelsize = 14)
#axs[0,4].set_ylim(0,1000)
# axs[0,1].set_ylabel(r'3D Flux power spectrum    [Mpc]', fontsize=16)
# and the fractional difference
for i in range(n):
    axs[1,0].fill_between(k_array, dif_mean_z1[:,i] - dif_error_z1[:,1], dif_mean_z1[:,i] + dif_error_z1[:,1], facecolor=dif_colors[i], edgecolor=dif_colors[i], label=dif_labels[i])
    axs[1,1].fill_between(k_array, dif_mean_z2[:,i] - dif_error_z2[:,1], dif_mean_z2[:,i] + dif_error_z2[:,1], facecolor=dif_colors[i], edgecolor=dif_colors[i], label=dif_labels[i])
    axs[1,2].fill_between(k_array, dif_mean_z3[:,i] - dif_error_z3[:,1], dif_mean_z3[:,i] + dif_error_z3[:,1], facecolor=dif_colors[i], edgecolor=dif_colors[i], label=dif_labels[i])
    axs[1,3].fill_between(k_array, dif_mean_z4[:,i] - dif_error_z4[:,1], dif_mean_z4[:,i] + dif_error_z4[:,1], facecolor=dif_colors[i], edgecolor=dif_colors[i], label=dif_labels[i])
    axs[1,4].fill_between(k_array, dif_mean_z5[:,i] - dif_error_z5[:,1], dif_mean_z5[:,i] + dif_error_z5[:,1], facecolor=dif_colors[i], edgecolor=dif_colors[i], label=dif_labels[i])
# axs[1,0].errorbar(k_array, dif_ave_ave_3keV_z1, yerr=dif_ave_ave_3keV_error_z1, fmt=':s', color='magenta', label=r'$m_{\rm wdm} = 3$ keV, $z=2$')
# axs[1,0].errorbar(k_array, dif_ave_ave_4keV_z1, yerr=dif_ave_ave_4keV_error_z1, fmt=':d', color='orange', label=r'$m_{\rm wdm} = 4$ keV, $z=2$')
# axs[1,0].errorbar(k_array, dif_ave_ave_6keV_z1, yerr=dif_ave_ave_6keV_error_z1, fmt='--p', color='purple', label=r'$m_{\rm wdm} = 6$ keV, $z=2$')
# axs[1,0].errorbar(k_array, dif_ave_ave_9keV_z1, yerr=dif_ave_ave_9keV_error_z1, fmt='-.*', color='green', label=r'$m_{\rm wdm} = 9$ keV, $z=2$')
axs[1,0].grid(linestyle='dotted')
axs[1,0].set_xscale('log')
axs[1,0].set_xlabel('k   [Mpc$^{-1}$]', fontsize=16)
axs[1,0].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[1,0].set_ylabel(r'Frac. diff.    [%]', fontsize=16)
axs[1,0].set_xlim(0.05, 1.1)
#axs[1,0].set_ylim(0., 2.6)
# axs[1,0].legend(loc='best',fontsize=14)
# and the other redshift
# axs[1,1].errorbar(k_array, dif_ave_ave_3keV_z5, yerr=dif_ave_ave_3keV_error_z5, fmt=':s', color='magenta', label=r'$m_{\rm wdm} = 3$ keV, $z=4$')
# axs[1,1].errorbar(k_array, dif_ave_ave_4keV_z5, yerr=dif_ave_ave_4keV_error_z5, fmt=':d', color='orange', label=r'$m_{\rm wdm} = 4$ keV, $z=4$')
# axs[1,1].errorbar(k_array, dif_ave_ave_6keV_z5, yerr=dif_ave_ave_6keV_error_z5, fmt='--p', color='purple', label=r'$m_{\rm wdm} = 6$ keV, $z=4$')
# axs[1,1].errorbar(k_array, dif_ave_ave_9keV_z5, yerr=dif_ave_ave_9keV_error_z5, fmt='-.*', color='green', label=r'$m_{\rm wdm} = 9$ keV,$z=4$')
axs[1,1].grid(linestyle='dotted')
axs[1,1].set_xscale('log')
axs[1,1].set_xlabel('k   [Mpc$^{-1}$]', fontsize=16)
axs[1,1].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[1,1].set_xlim(0.05, 1.1)
axs[1,2].grid(linestyle='dotted')
axs[1,2].set_xscale('log')
axs[1,2].set_xlabel('k   [Mpc$^{-1}$]', fontsize=16)
axs[1,2].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[1,2].set_xlim(0.05, 1.1)
axs[1,3].grid(linestyle='dotted')
axs[1,3].set_xscale('log')
axs[1,3].set_xlabel('k   [Mpc$^{-1}$]', fontsize=16)
axs[1,3].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[1,3].set_xlim(0.05, 1.1)
axs[1,4].grid(linestyle='dotted')
axs[1,4].set_xscale('log')
axs[1,4].set_xlabel('k   [Mpc$^{-1}$]', fontsize=16)
axs[1,4].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[1,4].set_xlim(0.05, 1.1)
#axs[1,1].set_ylim(0., 13)
fig.tight_layout()
plt.savefig('3D_power_w_errors.pdf')
plt.show()

"""
Note for myself: Results seems a little suspect, perhaps take a closer look at Catalina's data.
"""
"\nNote for myself: Results seems a little suspect, perhaps take a closer look at Catalina's data.\n"
# luckily for the transparency data only gadget sims matter, so
# now we want a transparency plot
z_re = np.linspace(6.0, 12.0, 30)
psi_ave_cdm_z1 = np.zeros(len(z_re))
psi_ave_3keV_z1 = np.zeros(len(z_re))
psi_ave_4keV_z1 = np.zeros(len(z_re))
psi_ave_6keV_z1 = np.zeros(len(z_re))
psi_ave_9keV_z1 = np.zeros(len(z_re))

psi_ave_cdm_z2 = np.zeros(len(z_re))
psi_ave_3keV_z2 = np.zeros(len(z_re))
psi_ave_4keV_z2 = np.zeros(len(z_re))
psi_ave_6keV_z2 = np.zeros(len(z_re))
psi_ave_9keV_z2 = np.zeros(len(z_re))

psi_ave_cdm_z3 = np.zeros(len(z_re))
psi_ave_3keV_z3 = np.zeros(len(z_re))
psi_ave_4keV_z3 = np.zeros(len(z_re))
psi_ave_6keV_z3 = np.zeros(len(z_re))
psi_ave_9keV_z3 = np.zeros(len(z_re))

psi_ave_cdm_z4 = np.zeros(len(z_re))
psi_ave_3keV_z4 = np.zeros(len(z_re))
psi_ave_4keV_z4 = np.zeros(len(z_re))
psi_ave_6keV_z4 = np.zeros(len(z_re))
psi_ave_9keV_z4 = np.zeros(len(z_re))

psi_ave_cdm_z5 = np.zeros(len(z_re))
psi_ave_3keV_z5 = np.zeros(len(z_re))
psi_ave_4keV_z5 = np.zeros(len(z_re))
psi_ave_6keV_z5 = np.zeros(len(z_re))
psi_ave_9keV_z5 = np.zeros(len(z_re))

psi_r1_cdm_z1 = np.zeros(len(z_re))
psi_r2_cdm_z1 = np.zeros(len(z_re))
psi_r3_cdm_z1 = np.zeros(len(z_re))
psi_r4_cdm_z1 = np.zeros(len(z_re))
psi_r5_cdm_z1 = np.zeros(len(z_re))
psi_r6_cdm_z1 = np.zeros(len(z_re))
psi_r7_cdm_z1 = np.zeros(len(z_re))
psi_r8_cdm_z1 = np.zeros(len(z_re))

psi_r1_3keV_z1 = np.zeros(len(z_re))
psi_r2_3keV_z1 = np.zeros(len(z_re))
psi_r3_3keV_z1 = np.zeros(len(z_re))
psi_r4_3keV_z1 = np.zeros(len(z_re))
psi_r5_3keV_z1 = np.zeros(len(z_re))
psi_r6_3keV_z1 = np.zeros(len(z_re))
psi_r7_3keV_z1 = np.zeros(len(z_re))
psi_r8_3keV_z1 = np.zeros(len(z_re))

psi_r1_4keV_z1 = np.zeros(len(z_re))
psi_r2_4keV_z1 = np.zeros(len(z_re))
psi_r3_4keV_z1 = np.zeros(len(z_re))
psi_r4_4keV_z1 = np.zeros(len(z_re))
psi_r5_4keV_z1 = np.zeros(len(z_re))
psi_r6_4keV_z1 = np.zeros(len(z_re))
psi_r7_4keV_z1 = np.zeros(len(z_re))
psi_r8_4keV_z1 = np.zeros(len(z_re))

psi_r1_6keV_z1 = np.zeros(len(z_re))
psi_r2_6keV_z1 = np.zeros(len(z_re))
psi_r3_6keV_z1 = np.zeros(len(z_re))
psi_r4_6keV_z1 = np.zeros(len(z_re))
psi_r5_6keV_z1 = np.zeros(len(z_re))
psi_r6_6keV_z1 = np.zeros(len(z_re))
psi_r7_6keV_z1 = np.zeros(len(z_re))
psi_r8_6keV_z1 = np.zeros(len(z_re))

psi_r1_9keV_z1 = np.zeros(len(z_re))
psi_r2_9keV_z1 = np.zeros(len(z_re))
psi_r3_9keV_z1 = np.zeros(len(z_re))
psi_r4_9keV_z1 = np.zeros(len(z_re))
psi_r5_9keV_z1 = np.zeros(len(z_re))
psi_r6_9keV_z1 = np.zeros(len(z_re))
psi_r7_9keV_z1 = np.zeros(len(z_re))
psi_r8_9keV_z1 = np.zeros(len(z_re))

psi_r1_cdm_z2 = np.zeros(len(z_re))
psi_r2_cdm_z2 = np.zeros(len(z_re))
psi_r3_cdm_z2 = np.zeros(len(z_re))
psi_r4_cdm_z2 = np.zeros(len(z_re))
psi_r5_cdm_z2 = np.zeros(len(z_re))
psi_r6_cdm_z2 = np.zeros(len(z_re))
psi_r7_cdm_z2 = np.zeros(len(z_re))
psi_r8_cdm_z2 = np.zeros(len(z_re))

psi_r1_3keV_z2 = np.zeros(len(z_re))
psi_r2_3keV_z2 = np.zeros(len(z_re))
psi_r3_3keV_z2 = np.zeros(len(z_re))
psi_r4_3keV_z2 = np.zeros(len(z_re))
psi_r5_3keV_z2 = np.zeros(len(z_re))
psi_r6_3keV_z2 = np.zeros(len(z_re))
psi_r7_3keV_z2 = np.zeros(len(z_re))
psi_r8_3keV_z2 = np.zeros(len(z_re))

psi_r1_4keV_z2 = np.zeros(len(z_re))
psi_r2_4keV_z2 = np.zeros(len(z_re))
psi_r3_4keV_z2 = np.zeros(len(z_re))
psi_r4_4keV_z2 = np.zeros(len(z_re))
psi_r5_4keV_z2 = np.zeros(len(z_re))
psi_r6_4keV_z2 = np.zeros(len(z_re))
psi_r7_4keV_z2 = np.zeros(len(z_re))
psi_r8_4keV_z2 = np.zeros(len(z_re))

psi_r1_6keV_z2 = np.zeros(len(z_re))
psi_r2_6keV_z2 = np.zeros(len(z_re))
psi_r3_6keV_z2 = np.zeros(len(z_re))
psi_r4_6keV_z2 = np.zeros(len(z_re))
psi_r5_6keV_z2 = np.zeros(len(z_re))
psi_r6_6keV_z2 = np.zeros(len(z_re))
psi_r7_6keV_z2 = np.zeros(len(z_re))
psi_r8_6keV_z2 = np.zeros(len(z_re))

psi_r1_9keV_z2 = np.zeros(len(z_re))
psi_r2_9keV_z2 = np.zeros(len(z_re))
psi_r3_9keV_z2 = np.zeros(len(z_re))
psi_r4_9keV_z2 = np.zeros(len(z_re))
psi_r5_9keV_z2 = np.zeros(len(z_re))
psi_r6_9keV_z2 = np.zeros(len(z_re))
psi_r7_9keV_z2 = np.zeros(len(z_re))
psi_r8_9keV_z2 = np.zeros(len(z_re))

psi_r1_cdm_z3 = np.zeros(len(z_re))
psi_r2_cdm_z3 = np.zeros(len(z_re))
psi_r3_cdm_z3 = np.zeros(len(z_re))
psi_r4_cdm_z3 = np.zeros(len(z_re))
psi_r5_cdm_z3 = np.zeros(len(z_re))
psi_r6_cdm_z3 = np.zeros(len(z_re))
psi_r7_cdm_z3 = np.zeros(len(z_re))
psi_r8_cdm_z3 = np.zeros(len(z_re))

psi_r1_3keV_z3 = np.zeros(len(z_re))
psi_r2_3keV_z3 = np.zeros(len(z_re))
psi_r3_3keV_z3 = np.zeros(len(z_re))
psi_r4_3keV_z3 = np.zeros(len(z_re))
psi_r5_3keV_z3 = np.zeros(len(z_re))
psi_r6_3keV_z3 = np.zeros(len(z_re))
psi_r7_3keV_z3 = np.zeros(len(z_re))
psi_r8_3keV_z3 = np.zeros(len(z_re))

psi_r1_4keV_z3 = np.zeros(len(z_re))
psi_r2_4keV_z3 = np.zeros(len(z_re))
psi_r3_4keV_z3 = np.zeros(len(z_re))
psi_r4_4keV_z3 = np.zeros(len(z_re))
psi_r5_4keV_z3 = np.zeros(len(z_re))
psi_r6_4keV_z3 = np.zeros(len(z_re))
psi_r7_4keV_z3 = np.zeros(len(z_re))
psi_r8_4keV_z3 = np.zeros(len(z_re))

psi_r1_6keV_z3 = np.zeros(len(z_re))
psi_r2_6keV_z3 = np.zeros(len(z_re))
psi_r3_6keV_z3 = np.zeros(len(z_re))
psi_r4_6keV_z3 = np.zeros(len(z_re))
psi_r5_6keV_z3 = np.zeros(len(z_re))
psi_r6_6keV_z3 = np.zeros(len(z_re))
psi_r7_6keV_z3 = np.zeros(len(z_re))
psi_r8_6keV_z3 = np.zeros(len(z_re))

psi_r1_9keV_z3 = np.zeros(len(z_re))
psi_r2_9keV_z3 = np.zeros(len(z_re))
psi_r3_9keV_z3 = np.zeros(len(z_re))
psi_r4_9keV_z3 = np.zeros(len(z_re))
psi_r5_9keV_z3 = np.zeros(len(z_re))
psi_r6_9keV_z3 = np.zeros(len(z_re))
psi_r7_9keV_z3 = np.zeros(len(z_re))
psi_r8_9keV_z3 = np.zeros(len(z_re))

psi_r1_cdm_z4 = np.zeros(len(z_re))
psi_r2_cdm_z4 = np.zeros(len(z_re))
psi_r3_cdm_z4 = np.zeros(len(z_re))
psi_r4_cdm_z4 = np.zeros(len(z_re))
psi_r5_cdm_z4 = np.zeros(len(z_re))
psi_r6_cdm_z4 = np.zeros(len(z_re))
psi_r7_cdm_z4 = np.zeros(len(z_re))
psi_r8_cdm_z4 = np.zeros(len(z_re))

psi_r1_3keV_z4 = np.zeros(len(z_re))
psi_r2_3keV_z4 = np.zeros(len(z_re))
psi_r3_3keV_z4 = np.zeros(len(z_re))
psi_r4_3keV_z4 = np.zeros(len(z_re))
psi_r5_3keV_z4 = np.zeros(len(z_re))
psi_r6_3keV_z4 = np.zeros(len(z_re))
psi_r7_3keV_z4 = np.zeros(len(z_re))
psi_r8_3keV_z4 = np.zeros(len(z_re))

psi_r1_4keV_z4 = np.zeros(len(z_re))
psi_r2_4keV_z4 = np.zeros(len(z_re))
psi_r3_4keV_z4 = np.zeros(len(z_re))
psi_r4_4keV_z4 = np.zeros(len(z_re))
psi_r5_4keV_z4 = np.zeros(len(z_re))
psi_r6_4keV_z4 = np.zeros(len(z_re))
psi_r7_4keV_z4 = np.zeros(len(z_re))
psi_r8_4keV_z4 = np.zeros(len(z_re))

psi_r1_6keV_z4 = np.zeros(len(z_re))
psi_r2_6keV_z4 = np.zeros(len(z_re))
psi_r3_6keV_z4 = np.zeros(len(z_re))
psi_r4_6keV_z4 = np.zeros(len(z_re))
psi_r5_6keV_z4 = np.zeros(len(z_re))
psi_r6_6keV_z4 = np.zeros(len(z_re))
psi_r7_6keV_z4 = np.zeros(len(z_re))
psi_r8_6keV_z4 = np.zeros(len(z_re))

psi_r1_9keV_z4 = np.zeros(len(z_re))
psi_r2_9keV_z4 = np.zeros(len(z_re))
psi_r3_9keV_z4 = np.zeros(len(z_re))
psi_r4_9keV_z4 = np.zeros(len(z_re))
psi_r5_9keV_z4 = np.zeros(len(z_re))
psi_r6_9keV_z4 = np.zeros(len(z_re))
psi_r7_9keV_z4 = np.zeros(len(z_re))
psi_r8_9keV_z4 = np.zeros(len(z_re))

psi_r1_cdm_z5 = np.zeros(len(z_re))
psi_r2_cdm_z5 = np.zeros(len(z_re))
psi_r3_cdm_z5 = np.zeros(len(z_re))
psi_r4_cdm_z5 = np.zeros(len(z_re))
psi_r5_cdm_z5 = np.zeros(len(z_re))
psi_r6_cdm_z5 = np.zeros(len(z_re))
psi_r7_cdm_z5 = np.zeros(len(z_re))
psi_r8_cdm_z5 = np.zeros(len(z_re))

psi_r1_3keV_z5 = np.zeros(len(z_re))
psi_r2_3keV_z5 = np.zeros(len(z_re))
psi_r3_3keV_z5 = np.zeros(len(z_re))
psi_r4_3keV_z5 = np.zeros(len(z_re))
psi_r5_3keV_z5 = np.zeros(len(z_re))
psi_r6_3keV_z5 = np.zeros(len(z_re))
psi_r7_3keV_z5 = np.zeros(len(z_re))
psi_r8_3keV_z5 = np.zeros(len(z_re))

psi_r1_4keV_z5 = np.zeros(len(z_re))
psi_r2_4keV_z5 = np.zeros(len(z_re))
psi_r3_4keV_z5 = np.zeros(len(z_re))
psi_r4_4keV_z5 = np.zeros(len(z_re))
psi_r5_4keV_z5 = np.zeros(len(z_re))
psi_r6_4keV_z5 = np.zeros(len(z_re))
psi_r7_4keV_z5 = np.zeros(len(z_re))
psi_r8_4keV_z5 = np.zeros(len(z_re))

psi_r1_6keV_z5 = np.zeros(len(z_re))
psi_r2_6keV_z5 = np.zeros(len(z_re))
psi_r3_6keV_z5 = np.zeros(len(z_re))
psi_r4_6keV_z5 = np.zeros(len(z_re))
psi_r5_6keV_z5 = np.zeros(len(z_re))
psi_r6_6keV_z5 = np.zeros(len(z_re))
psi_r7_6keV_z5 = np.zeros(len(z_re))
psi_r8_6keV_z5 = np.zeros(len(z_re))

psi_r1_9keV_z5 = np.zeros(len(z_re))
psi_r2_9keV_z5 = np.zeros(len(z_re))
psi_r3_9keV_z5 = np.zeros(len(z_re))
psi_r4_9keV_z5 = np.zeros(len(z_re))
psi_r5_9keV_z5 = np.zeros(len(z_re))
psi_r6_9keV_z5 = np.zeros(len(z_re))
psi_r7_9keV_z5 = np.zeros(len(z_re))
psi_r8_9keV_z5 = np.zeros(len(z_re))

for i in range(0, len(z_re)):
    psi_ave_cdm_z1[i] = ave_ave_cdm.transparency(z_re[i], 2.0)
    psi_ave_3keV_z1[i] = ave_ave_3keV.transparency(z_re[i], 2.0)
    psi_ave_4keV_z1[i] = ave_ave_4keV.transparency(z_re[i], 2.0)
    psi_ave_6keV_z1[i] = ave_ave_6keV.transparency(z_re[i], 2.0)
    psi_ave_9keV_z1[i] = ave_ave_9keV.transparency(z_re[i], 2.0)
    
    psi_ave_cdm_z2[i] = ave_ave_cdm.transparency(z_re[i], 2.5)
    psi_ave_3keV_z2[i] = ave_ave_3keV.transparency(z_re[i], 2.5)
    psi_ave_4keV_z2[i] = ave_ave_4keV.transparency(z_re[i], 2.5)
    psi_ave_6keV_z2[i] = ave_ave_6keV.transparency(z_re[i], 2.5)
    psi_ave_9keV_z2[i] = ave_ave_9keV.transparency(z_re[i], 2.5)

    psi_ave_cdm_z3[i]  = ave_ave_cdm.transparency(z_re[i], 3.0)
    psi_ave_3keV_z3[i] = ave_ave_3keV.transparency(z_re[i], 3.0)
    psi_ave_4keV_z3[i] = ave_ave_4keV.transparency(z_re[i], 3.0)
    psi_ave_6keV_z3[i] = ave_ave_6keV.transparency(z_re[i], 3.0)
    psi_ave_9keV_z3[i] = ave_ave_9keV.transparency(z_re[i], 3.0)

    psi_ave_cdm_z4[i] = ave_ave_cdm.transparency(z_re[i], 3.5)
    psi_ave_3keV_z4[i] = ave_ave_3keV.transparency(z_re[i], 3.5)
    psi_ave_4keV_z4[i] = ave_ave_4keV.transparency(z_re[i], 3.5)
    psi_ave_6keV_z4[i] = ave_ave_6keV.transparency(z_re[i], 3.5)
    psi_ave_9keV_z4[i] = ave_ave_9keV.transparency(z_re[i], 3.5)

    psi_ave_cdm_z5[i] = ave_ave_cdm.transparency(z_re[i], 4.0)
    psi_ave_3keV_z5[i] = ave_ave_3keV.transparency(z_re[i], 4.0)
    psi_ave_4keV_z5[i] = ave_ave_4keV.transparency(z_re[i], 4.0)
    psi_ave_6keV_z5[i] = ave_ave_6keV.transparency(z_re[i], 4.0)
    psi_ave_9keV_z5[i] = ave_ave_9keV.transparency(z_re[i], 4.0)
    
    psi_r1_cdm_z1[i] = ave_r1_cdm.transparency(z_re[i], z1)
    psi_r2_cdm_z1[i] = ave_r2_cdm.transparency(z_re[i], z1)
    psi_r3_cdm_z1[i] = ave_r3_cdm.transparency(z_re[i], z1)
    psi_r4_cdm_z1[i] = ave_r4_cdm.transparency(z_re[i], z1)
    psi_r5_cdm_z1[i] = ave_r5_cdm.transparency(z_re[i], z1)
    psi_r6_cdm_z1[i] = ave_r6_cdm.transparency(z_re[i], z1)
    psi_r7_cdm_z1[i] = ave_r7_cdm.transparency(z_re[i], z1)
    psi_r8_cdm_z1[i] = ave_r8_cdm.transparency(z_re[i], z1)
    
    psi_r1_3keV_z1[i] = ave_r1_3keV.transparency(z_re[i], z1)
    psi_r2_3keV_z1[i] = ave_r2_3keV.transparency(z_re[i], z1)
    psi_r3_3keV_z1[i] = ave_r3_3keV.transparency(z_re[i], z1)
    psi_r4_3keV_z1[i] = ave_r4_3keV.transparency(z_re[i], z1)
    psi_r5_3keV_z1[i] = ave_r5_3keV.transparency(z_re[i], z1)
    psi_r6_3keV_z1[i] = ave_r6_3keV.transparency(z_re[i], z1)
    psi_r7_3keV_z1[i] = ave_r7_3keV.transparency(z_re[i], z1)
    psi_r8_3keV_z1[i] = ave_r8_3keV.transparency(z_re[i], z1)  
    
 
    psi_r1_4keV_z1[i] = ave_r1_4keV.transparency(z_re[i], z1)
    psi_r2_4keV_z1[i] = ave_r2_4keV.transparency(z_re[i], z1)
    psi_r3_4keV_z1[i] = ave_r3_4keV.transparency(z_re[i], z1)
    psi_r4_4keV_z1[i] = ave_r4_4keV.transparency(z_re[i], z1)
    psi_r5_4keV_z1[i] = ave_r5_4keV.transparency(z_re[i], z1)
    psi_r6_4keV_z1[i] = ave_r6_4keV.transparency(z_re[i], z1)
    psi_r7_4keV_z1[i] = ave_r7_4keV.transparency(z_re[i], z1)
    psi_r8_4keV_z1[i] = ave_r8_4keV.transparency(z_re[i], z1)    
    
    psi_r1_6keV_z1[i] = ave_r1_6keV.transparency(z_re[i], z1)
    psi_r2_6keV_z1[i] = ave_r2_6keV.transparency(z_re[i], z1)
    psi_r3_6keV_z1[i] = ave_r3_6keV.transparency(z_re[i], z1)
    psi_r4_6keV_z1[i] = ave_r4_6keV.transparency(z_re[i], z1)
    psi_r5_6keV_z1[i] = ave_r5_6keV.transparency(z_re[i], z1)
    psi_r6_6keV_z1[i] = ave_r6_6keV.transparency(z_re[i], z1)
    psi_r7_6keV_z1[i] = ave_r7_6keV.transparency(z_re[i], z1)
    psi_r8_6keV_z1[i] = ave_r8_6keV.transparency(z_re[i], z1)
    
    psi_r1_9keV_z1[i] = ave_r1_9keV.transparency(z_re[i], z1)
    psi_r2_9keV_z1[i] = ave_r2_9keV.transparency(z_re[i], z1)
    psi_r3_9keV_z1[i] = ave_r3_9keV.transparency(z_re[i], z1)
    psi_r4_9keV_z1[i] = ave_r4_9keV.transparency(z_re[i], z1)
    psi_r5_9keV_z1[i] = ave_r5_9keV.transparency(z_re[i], z1)
    psi_r6_9keV_z1[i] = ave_r6_9keV.transparency(z_re[i], z1)
    psi_r7_9keV_z1[i] = ave_r7_9keV.transparency(z_re[i], z1)
    psi_r8_9keV_z1[i] = ave_r8_9keV.transparency(z_re[i], z1)

    # z2
    psi_r1_cdm_z2[i] = ave_r1_cdm.transparency(z_re[i], z2)
    psi_r2_cdm_z2[i] = ave_r2_cdm.transparency(z_re[i], z2)
    psi_r3_cdm_z2[i] = ave_r3_cdm.transparency(z_re[i], z2)
    psi_r4_cdm_z2[i] = ave_r4_cdm.transparency(z_re[i], z2)
    psi_r5_cdm_z2[i] = ave_r5_cdm.transparency(z_re[i], z2)
    psi_r6_cdm_z2[i] = ave_r6_cdm.transparency(z_re[i], z2)
    psi_r7_cdm_z2[i] = ave_r7_cdm.transparency(z_re[i], z2)
    psi_r8_cdm_z2[i] = ave_r8_cdm.transparency(z_re[i], z2)
    
    psi_r1_3keV_z2[i] = ave_r1_3keV.transparency(z_re[i], z2)
    psi_r2_3keV_z2[i] = ave_r2_3keV.transparency(z_re[i], z2)
    psi_r3_3keV_z2[i] = ave_r3_3keV.transparency(z_re[i], z2)
    psi_r4_3keV_z2[i] = ave_r4_3keV.transparency(z_re[i], z2)
    psi_r5_3keV_z2[i] = ave_r5_3keV.transparency(z_re[i], z2)
    psi_r6_3keV_z2[i] = ave_r6_3keV.transparency(z_re[i], z2)
    psi_r7_3keV_z2[i] = ave_r7_3keV.transparency(z_re[i], z2)
    psi_r8_3keV_z2[i] = ave_r8_3keV.transparency(z_re[i], z2)  
    
 
    psi_r1_4keV_z2[i] = ave_r1_4keV.transparency(z_re[i], z2)
    psi_r2_4keV_z2[i] = ave_r2_4keV.transparency(z_re[i], z2)
    psi_r3_4keV_z2[i] = ave_r3_4keV.transparency(z_re[i], z2)
    psi_r4_4keV_z2[i] = ave_r4_4keV.transparency(z_re[i], z2)
    psi_r5_4keV_z2[i] = ave_r5_4keV.transparency(z_re[i], z2)
    psi_r6_4keV_z2[i] = ave_r6_4keV.transparency(z_re[i], z2)
    psi_r7_4keV_z2[i] = ave_r7_4keV.transparency(z_re[i], z2)
    psi_r8_4keV_z2[i] = ave_r8_4keV.transparency(z_re[i], z2)    
    
    psi_r1_6keV_z2[i] = ave_r1_6keV.transparency(z_re[i], z2)
    psi_r2_6keV_z2[i] = ave_r2_6keV.transparency(z_re[i], z2)
    psi_r3_6keV_z2[i] = ave_r3_6keV.transparency(z_re[i], z2)
    psi_r4_6keV_z2[i] = ave_r4_6keV.transparency(z_re[i], z2)
    psi_r5_6keV_z2[i] = ave_r5_6keV.transparency(z_re[i], z2)
    psi_r6_6keV_z2[i] = ave_r6_6keV.transparency(z_re[i], z2)
    psi_r7_6keV_z2[i] = ave_r7_6keV.transparency(z_re[i], z2)
    psi_r8_6keV_z2[i] = ave_r8_6keV.transparency(z_re[i], z2)
    
    psi_r1_9keV_z2[i] = ave_r1_9keV.transparency(z_re[i], z2)
    psi_r2_9keV_z2[i] = ave_r2_9keV.transparency(z_re[i], z2)
    psi_r3_9keV_z2[i] = ave_r3_9keV.transparency(z_re[i], z2)
    psi_r4_9keV_z2[i] = ave_r4_9keV.transparency(z_re[i], z2)
    psi_r5_9keV_z2[i] = ave_r5_9keV.transparency(z_re[i], z2)
    psi_r6_9keV_z2[i] = ave_r6_9keV.transparency(z_re[i], z2)
    psi_r7_9keV_z2[i] = ave_r7_9keV.transparency(z_re[i], z2)
    psi_r8_9keV_z2[i] = ave_r8_9keV.transparency(z_re[i], z2)
    
    #z3
    psi_r1_cdm_z3[i] = ave_r1_cdm.transparency(z_re[i], z3)
    psi_r2_cdm_z3[i] = ave_r2_cdm.transparency(z_re[i], z3)
    psi_r3_cdm_z3[i] = ave_r3_cdm.transparency(z_re[i], z3)
    psi_r4_cdm_z3[i] = ave_r4_cdm.transparency(z_re[i], z3)
    psi_r5_cdm_z3[i] = ave_r5_cdm.transparency(z_re[i], z3)
    psi_r6_cdm_z3[i] = ave_r6_cdm.transparency(z_re[i], z3)
    psi_r7_cdm_z3[i] = ave_r7_cdm.transparency(z_re[i], z3)
    psi_r8_cdm_z3[i] = ave_r8_cdm.transparency(z_re[i], z3)
    
    psi_r1_3keV_z3[i] = ave_r1_3keV.transparency(z_re[i], z3)
    psi_r2_3keV_z3[i] = ave_r2_3keV.transparency(z_re[i], z3)
    psi_r3_3keV_z3[i] = ave_r3_3keV.transparency(z_re[i], z3)
    psi_r4_3keV_z3[i] = ave_r4_3keV.transparency(z_re[i], z3)
    psi_r5_3keV_z3[i] = ave_r5_3keV.transparency(z_re[i], z3)
    psi_r6_3keV_z3[i] = ave_r6_3keV.transparency(z_re[i], z3)
    psi_r7_3keV_z3[i] = ave_r7_3keV.transparency(z_re[i], z3)
    psi_r8_3keV_z3[i] = ave_r8_3keV.transparency(z_re[i], z3)  
    
 
    psi_r1_4keV_z3[i] = ave_r1_4keV.transparency(z_re[i], z3)
    psi_r2_4keV_z3[i] = ave_r2_4keV.transparency(z_re[i], z3)
    psi_r3_4keV_z3[i] = ave_r3_4keV.transparency(z_re[i], z3)
    psi_r4_4keV_z3[i] = ave_r4_4keV.transparency(z_re[i], z3)
    psi_r5_4keV_z3[i] = ave_r5_4keV.transparency(z_re[i], z3)
    psi_r6_4keV_z3[i] = ave_r6_4keV.transparency(z_re[i], z3)
    psi_r7_4keV_z3[i] = ave_r7_4keV.transparency(z_re[i], z3)
    psi_r8_4keV_z3[i] = ave_r8_4keV.transparency(z_re[i], z3)    
    
    psi_r1_6keV_z3[i] = ave_r1_6keV.transparency(z_re[i], z3)
    psi_r2_6keV_z3[i] = ave_r2_6keV.transparency(z_re[i], z3)
    psi_r3_6keV_z3[i] = ave_r3_6keV.transparency(z_re[i], z3)
    psi_r4_6keV_z3[i] = ave_r4_6keV.transparency(z_re[i], z3)
    psi_r5_6keV_z3[i] = ave_r5_6keV.transparency(z_re[i], z3)
    psi_r6_6keV_z3[i] = ave_r6_6keV.transparency(z_re[i], z3)
    psi_r7_6keV_z3[i] = ave_r7_6keV.transparency(z_re[i], z3)
    psi_r8_6keV_z3[i] = ave_r8_6keV.transparency(z_re[i], z3)
    
    psi_r1_9keV_z3[i] = ave_r1_9keV.transparency(z_re[i], z3)
    psi_r2_9keV_z3[i] = ave_r2_9keV.transparency(z_re[i], z3)
    psi_r3_9keV_z3[i] = ave_r3_9keV.transparency(z_re[i], z3)
    psi_r4_9keV_z3[i] = ave_r4_9keV.transparency(z_re[i], z3)
    psi_r5_9keV_z3[i] = ave_r5_9keV.transparency(z_re[i], z3)
    psi_r6_9keV_z3[i] = ave_r6_9keV.transparency(z_re[i], z3)
    psi_r7_9keV_z3[i] = ave_r7_9keV.transparency(z_re[i], z3)
    psi_r8_9keV_z3[i] = ave_r8_9keV.transparency(z_re[i], z3)

    # z4
    psi_r1_cdm_z4[i] = ave_r1_cdm.transparency(z_re[i], z4)
    psi_r2_cdm_z4[i] = ave_r2_cdm.transparency(z_re[i], z4)
    psi_r3_cdm_z4[i] = ave_r3_cdm.transparency(z_re[i], z4)
    psi_r4_cdm_z4[i] = ave_r4_cdm.transparency(z_re[i], z4)
    psi_r5_cdm_z4[i] = ave_r5_cdm.transparency(z_re[i], z4)
    psi_r6_cdm_z4[i] = ave_r6_cdm.transparency(z_re[i], z4)
    psi_r7_cdm_z4[i] = ave_r7_cdm.transparency(z_re[i], z4)
    psi_r8_cdm_z4[i] = ave_r8_cdm.transparency(z_re[i], z4)
    
    psi_r1_3keV_z4[i] = ave_r1_3keV.transparency(z_re[i], z4)
    psi_r2_3keV_z4[i] = ave_r2_3keV.transparency(z_re[i], z4)
    psi_r3_3keV_z4[i] = ave_r3_3keV.transparency(z_re[i], z4)
    psi_r4_3keV_z4[i] = ave_r4_3keV.transparency(z_re[i], z4)
    psi_r5_3keV_z4[i] = ave_r5_3keV.transparency(z_re[i], z4)
    psi_r6_3keV_z4[i] = ave_r6_3keV.transparency(z_re[i], z4)
    psi_r7_3keV_z4[i] = ave_r7_3keV.transparency(z_re[i], z4)
    psi_r8_3keV_z4[i] = ave_r8_3keV.transparency(z_re[i], z4)  
    
 
    psi_r1_4keV_z4[i] = ave_r1_4keV.transparency(z_re[i], z4)
    psi_r2_4keV_z4[i] = ave_r2_4keV.transparency(z_re[i], z4)
    psi_r3_4keV_z4[i] = ave_r3_4keV.transparency(z_re[i], z4)
    psi_r4_4keV_z4[i] = ave_r4_4keV.transparency(z_re[i], z4)
    psi_r5_4keV_z4[i] = ave_r5_4keV.transparency(z_re[i], z4)
    psi_r6_4keV_z4[i] = ave_r6_4keV.transparency(z_re[i], z4)
    psi_r7_4keV_z4[i] = ave_r7_4keV.transparency(z_re[i], z4)
    psi_r8_4keV_z4[i] = ave_r8_4keV.transparency(z_re[i], z4)    
    
    psi_r1_6keV_z4[i] = ave_r1_6keV.transparency(z_re[i], z4)
    psi_r2_6keV_z4[i] = ave_r2_6keV.transparency(z_re[i], z4)
    psi_r3_6keV_z4[i] = ave_r3_6keV.transparency(z_re[i], z4)
    psi_r4_6keV_z4[i] = ave_r4_6keV.transparency(z_re[i], z4)
    psi_r5_6keV_z4[i] = ave_r5_6keV.transparency(z_re[i], z4)
    psi_r6_6keV_z4[i] = ave_r6_6keV.transparency(z_re[i], z4)
    psi_r7_6keV_z4[i] = ave_r7_6keV.transparency(z_re[i], z4)
    psi_r8_6keV_z4[i] = ave_r8_6keV.transparency(z_re[i], z4)
    
    psi_r1_9keV_z4[i] = ave_r1_9keV.transparency(z_re[i], z4)
    psi_r2_9keV_z4[i] = ave_r2_9keV.transparency(z_re[i], z4)
    psi_r3_9keV_z4[i] = ave_r3_9keV.transparency(z_re[i], z4)
    psi_r4_9keV_z4[i] = ave_r4_9keV.transparency(z_re[i], z4)
    psi_r5_9keV_z4[i] = ave_r5_9keV.transparency(z_re[i], z4)
    psi_r6_9keV_z4[i] = ave_r6_9keV.transparency(z_re[i], z4)
    psi_r7_9keV_z4[i] = ave_r7_9keV.transparency(z_re[i], z4)
    psi_r8_9keV_z4[i] = ave_r8_9keV.transparency(z_re[i], z4)

    # z5
    psi_r1_cdm_z5[i] = ave_r1_cdm.transparency(z_re[i], z5)
    psi_r2_cdm_z5[i] = ave_r2_cdm.transparency(z_re[i], z5)
    psi_r3_cdm_z5[i] = ave_r3_cdm.transparency(z_re[i], z5)
    psi_r4_cdm_z5[i] = ave_r4_cdm.transparency(z_re[i], z5)
    psi_r5_cdm_z5[i] = ave_r5_cdm.transparency(z_re[i], z5)
    psi_r6_cdm_z5[i] = ave_r6_cdm.transparency(z_re[i], z5)
    psi_r7_cdm_z5[i] = ave_r7_cdm.transparency(z_re[i], z5)
    psi_r8_cdm_z5[i] = ave_r8_cdm.transparency(z_re[i], z5)
    
    psi_r1_3keV_z5[i] = ave_r1_3keV.transparency(z_re[i], z5)
    psi_r2_3keV_z5[i] = ave_r2_3keV.transparency(z_re[i], z5)
    psi_r3_3keV_z5[i] = ave_r3_3keV.transparency(z_re[i], z5)
    psi_r4_3keV_z5[i] = ave_r4_3keV.transparency(z_re[i], z5)
    psi_r5_3keV_z5[i] = ave_r5_3keV.transparency(z_re[i], z5)
    psi_r6_3keV_z5[i] = ave_r6_3keV.transparency(z_re[i], z5)
    psi_r7_3keV_z5[i] = ave_r7_3keV.transparency(z_re[i], z5)
    psi_r8_3keV_z5[i] = ave_r8_3keV.transparency(z_re[i], z5)  
    
 
    psi_r1_4keV_z5[i] = ave_r1_4keV.transparency(z_re[i], z5)
    psi_r2_4keV_z5[i] = ave_r2_4keV.transparency(z_re[i], z5)
    psi_r3_4keV_z5[i] = ave_r3_4keV.transparency(z_re[i], z5)
    psi_r4_4keV_z5[i] = ave_r4_4keV.transparency(z_re[i], z5)
    psi_r5_4keV_z5[i] = ave_r5_4keV.transparency(z_re[i], z5)
    psi_r6_4keV_z5[i] = ave_r6_4keV.transparency(z_re[i], z5)
    psi_r7_4keV_z5[i] = ave_r7_4keV.transparency(z_re[i], z5)
    psi_r8_4keV_z5[i] = ave_r8_4keV.transparency(z_re[i], z5)    
    
    psi_r1_6keV_z5[i] = ave_r1_6keV.transparency(z_re[i], z5)
    psi_r2_6keV_z5[i] = ave_r2_6keV.transparency(z_re[i], z5)
    psi_r3_6keV_z5[i] = ave_r3_6keV.transparency(z_re[i], z5)
    psi_r4_6keV_z5[i] = ave_r4_6keV.transparency(z_re[i], z5)
    psi_r5_6keV_z5[i] = ave_r5_6keV.transparency(z_re[i], z5)
    psi_r6_6keV_z5[i] = ave_r6_6keV.transparency(z_re[i], z5)
    psi_r7_6keV_z5[i] = ave_r7_6keV.transparency(z_re[i], z5)
    psi_r8_6keV_z5[i] = ave_r8_6keV.transparency(z_re[i], z5)
    
    psi_r1_9keV_z5[i] = ave_r1_9keV.transparency(z_re[i], z5)
    psi_r2_9keV_z5[i] = ave_r2_9keV.transparency(z_re[i], z5)
    psi_r3_9keV_z5[i] = ave_r3_9keV.transparency(z_re[i], z5)
    psi_r4_9keV_z5[i] = ave_r4_9keV.transparency(z_re[i], z5)
    psi_r5_9keV_z5[i] = ave_r5_9keV.transparency(z_re[i], z5)
    psi_r6_9keV_z5[i] = ave_r6_9keV.transparency(z_re[i], z5)
    psi_r7_9keV_z5[i] = ave_r7_9keV.transparency(z_re[i], z5)
    psi_r8_9keV_z5[i] = ave_r8_9keV.transparency(z_re[i], z5)
    
    
def sample_variance_only_gadget(ave, r1, r2, r3, r4, r5, r6, r7, r8):
    "Need it for transparency stuff"
    temp = (r1 - ave)**2 + (r2 - ave)**2 + (r3 - ave)**2 + (r4 - ave)**2 + (r5 - ave)**2 + (r6 - ave)**2 + (r7 - ave)**2 + (r8 - ave)**2
    temp = np.sqrt(temp / 8.)
    return temp
# compute the errorbar
psi_error_cdm_z1 = sample_variance_only_gadget(psi_ave_cdm_z1, psi_r1_cdm_z1, psi_r2_cdm_z1, psi_r3_cdm_z1, psi_r4_cdm_z1, psi_r5_cdm_z1, psi_r6_cdm_z1, psi_r7_cdm_z1, psi_r8_cdm_z1)
psi_error_cdm_z2 = sample_variance_only_gadget(psi_ave_cdm_z2, psi_r1_cdm_z2, psi_r2_cdm_z2, psi_r3_cdm_z2, psi_r4_cdm_z2, psi_r5_cdm_z2, psi_r6_cdm_z2, psi_r7_cdm_z2, psi_r8_cdm_z2)
psi_error_cdm_z3 = sample_variance_only_gadget(psi_ave_cdm_z3, psi_r1_cdm_z3, psi_r2_cdm_z3, psi_r3_cdm_z3, psi_r4_cdm_z3, psi_r5_cdm_z3, psi_r6_cdm_z3, psi_r7_cdm_z3, psi_r8_cdm_z3)
psi_error_cdm_z4 = sample_variance_only_gadget(psi_ave_cdm_z4, psi_r1_cdm_z4, psi_r2_cdm_z4, psi_r3_cdm_z4, psi_r4_cdm_z4, psi_r5_cdm_z4, psi_r6_cdm_z4, psi_r7_cdm_z4, psi_r8_cdm_z4)
psi_error_cdm_z5 = sample_variance_only_gadget(psi_ave_cdm_z5, psi_r1_cdm_z5, psi_r2_cdm_z5, psi_r3_cdm_z5, psi_r4_cdm_z5, psi_r5_cdm_z5, psi_r6_cdm_z5, psi_r7_cdm_z5, psi_r8_cdm_z5)

psi_error_3keV_z1 = sample_variance_only_gadget(psi_ave_3keV_z1, psi_r1_3keV_z1, psi_r2_3keV_z1, psi_r3_3keV_z1, psi_r4_3keV_z1, psi_r5_3keV_z1, psi_r6_3keV_z1, psi_r7_3keV_z1, psi_r8_3keV_z1)
psi_error_4keV_z1 = sample_variance_only_gadget(psi_ave_4keV_z1, psi_r1_4keV_z1, psi_r2_4keV_z1, psi_r3_4keV_z1, psi_r4_4keV_z1, psi_r5_4keV_z1, psi_r6_4keV_z1, psi_r7_4keV_z1, psi_r8_4keV_z1)
psi_error_6keV_z1 = sample_variance_only_gadget(psi_ave_6keV_z1, psi_r1_6keV_z1, psi_r2_6keV_z1, psi_r3_6keV_z1, psi_r4_6keV_z1, psi_r5_6keV_z1, psi_r6_6keV_z1, psi_r7_6keV_z1, psi_r8_6keV_z1)
psi_error_9keV_z1 = sample_variance_only_gadget(psi_ave_9keV_z1, psi_r1_9keV_z1, psi_r2_9keV_z1, psi_r3_9keV_z1, psi_r4_9keV_z1, psi_r5_9keV_z1, psi_r6_9keV_z1, psi_r7_9keV_z1, psi_r8_9keV_z1)

psi_error_3keV_z2 = sample_variance_only_gadget(psi_ave_3keV_z2, psi_r1_3keV_z2, psi_r2_3keV_z2, psi_r3_3keV_z2, psi_r4_3keV_z2, psi_r5_3keV_z2, psi_r6_3keV_z2, psi_r7_3keV_z2, psi_r8_3keV_z2)
psi_error_4keV_z2 = sample_variance_only_gadget(psi_ave_4keV_z2, psi_r1_4keV_z2, psi_r2_4keV_z2, psi_r3_4keV_z2, psi_r4_4keV_z2, psi_r5_4keV_z2, psi_r6_4keV_z2, psi_r7_4keV_z2, psi_r8_4keV_z2)
psi_error_6keV_z2 = sample_variance_only_gadget(psi_ave_6keV_z2, psi_r1_6keV_z2, psi_r2_6keV_z2, psi_r3_6keV_z2, psi_r4_6keV_z2, psi_r5_6keV_z2, psi_r6_6keV_z2, psi_r7_6keV_z2, psi_r8_6keV_z2)
psi_error_9keV_z2 = sample_variance_only_gadget(psi_ave_9keV_z2, psi_r1_9keV_z2, psi_r2_9keV_z2, psi_r3_9keV_z2, psi_r4_9keV_z2, psi_r5_9keV_z2, psi_r6_9keV_z2, psi_r7_9keV_z2, psi_r8_9keV_z2)

psi_error_3keV_z3 = sample_variance_only_gadget(psi_ave_3keV_z3, psi_r1_3keV_z3, psi_r2_3keV_z3, psi_r3_3keV_z3, psi_r4_3keV_z3, psi_r5_3keV_z3, psi_r6_3keV_z3, psi_r7_3keV_z3, psi_r8_3keV_z3)
psi_error_4keV_z3 = sample_variance_only_gadget(psi_ave_4keV_z3, psi_r1_4keV_z3, psi_r2_4keV_z3, psi_r3_4keV_z3, psi_r4_4keV_z3, psi_r5_4keV_z3, psi_r6_4keV_z3, psi_r7_4keV_z3, psi_r8_4keV_z3)
psi_error_6keV_z3 = sample_variance_only_gadget(psi_ave_6keV_z3, psi_r1_6keV_z3, psi_r2_6keV_z3, psi_r3_6keV_z3, psi_r4_6keV_z3, psi_r5_6keV_z3, psi_r6_6keV_z3, psi_r7_6keV_z3, psi_r8_6keV_z3)
psi_error_9keV_z3 = sample_variance_only_gadget(psi_ave_9keV_z3, psi_r1_9keV_z3, psi_r2_9keV_z3, psi_r3_9keV_z3, psi_r4_9keV_z3, psi_r5_9keV_z3, psi_r6_9keV_z3, psi_r7_9keV_z3, psi_r8_9keV_z3)

psi_error_3keV_z4 = sample_variance_only_gadget(psi_ave_3keV_z4, psi_r1_3keV_z4, psi_r2_3keV_z4, psi_r3_3keV_z4, psi_r4_3keV_z4, psi_r5_3keV_z4, psi_r6_3keV_z4, psi_r7_3keV_z4, psi_r8_3keV_z4)
psi_error_4keV_z4 = sample_variance_only_gadget(psi_ave_4keV_z4, psi_r1_4keV_z4, psi_r2_4keV_z4, psi_r3_4keV_z4, psi_r4_4keV_z4, psi_r5_4keV_z4, psi_r6_4keV_z4, psi_r7_4keV_z4, psi_r8_4keV_z4)
psi_error_6keV_z4 = sample_variance_only_gadget(psi_ave_6keV_z4, psi_r1_6keV_z4, psi_r2_6keV_z4, psi_r3_6keV_z4, psi_r4_6keV_z4, psi_r5_6keV_z4, psi_r6_6keV_z4, psi_r7_6keV_z4, psi_r8_6keV_z4)
psi_error_9keV_z4 = sample_variance_only_gadget(psi_ave_9keV_z4, psi_r1_9keV_z4, psi_r2_9keV_z4, psi_r3_9keV_z4, psi_r4_9keV_z4, psi_r5_9keV_z4, psi_r6_9keV_z4, psi_r7_9keV_z4, psi_r8_9keV_z4)

psi_error_3keV_z5 = sample_variance_only_gadget(psi_ave_3keV_z5, psi_r1_3keV_z5, psi_r2_3keV_z5, psi_r3_3keV_z5, psi_r4_3keV_z5, psi_r5_3keV_z5, psi_r6_3keV_z5, psi_r7_3keV_z5, psi_r8_3keV_z5)
psi_error_4keV_z5 = sample_variance_only_gadget(psi_ave_4keV_z5, psi_r1_4keV_z5, psi_r2_4keV_z5, psi_r3_4keV_z5, psi_r4_4keV_z5, psi_r5_4keV_z5, psi_r6_4keV_z5, psi_r7_4keV_z5, psi_r8_4keV_z5)
psi_error_6keV_z5 = sample_variance_only_gadget(psi_ave_6keV_z5, psi_r1_6keV_z5, psi_r2_6keV_z5, psi_r3_6keV_z5, psi_r4_6keV_z5, psi_r5_6keV_z5, psi_r6_6keV_z5, psi_r7_6keV_z5, psi_r8_6keV_z5)
psi_error_9keV_z5 = sample_variance_only_gadget(psi_ave_9keV_z5, psi_r1_9keV_z5, psi_r2_9keV_z5, psi_r3_9keV_z5, psi_r4_9keV_z5, psi_r5_9keV_z5, psi_r6_9keV_z5, psi_r7_9keV_z5, psi_r8_9keV_z5)
mean_psi_z1 = np.zeros((len(z_re),n))
mean_psi_z2 = np.zeros((len(z_re),n))
mean_psi_z3 = np.zeros((len(z_re),n))
mean_psi_z4 = np.zeros((len(z_re),n))
mean_psi_z5 = np.zeros((len(z_re),n))
error_psi_z1 = np.zeros((len(z_re),n))
error_psi_z2 = np.zeros((len(z_re),n))
error_psi_z3 = np.zeros((len(z_re),n))
error_psi_z4 = np.zeros((len(z_re),n))
error_psi_z5 = np.zeros((len(z_re),n))

mean_psi_z1[:,0] =  psi_ave_cdm_z1[:]
mean_psi_z1[:,1] = psi_ave_9keV_z1[:]
mean_psi_z1[:,2] = psi_ave_6keV_z1[:]
mean_psi_z1[:,3] = psi_ave_4keV_z1[:]
mean_psi_z1[:,4] = psi_ave_3keV_z1[:]

mean_psi_z2[:,0] =  psi_ave_cdm_z2[:]
mean_psi_z2[:,1] = psi_ave_9keV_z2[:]
mean_psi_z2[:,2] = psi_ave_6keV_z2[:]
mean_psi_z2[:,3] = psi_ave_4keV_z2[:]
mean_psi_z2[:,4] = psi_ave_3keV_z2[:]

mean_psi_z3[:,0] =  psi_ave_cdm_z3[:]
mean_psi_z3[:,1] = psi_ave_9keV_z3[:]
mean_psi_z3[:,2] = psi_ave_6keV_z3[:]
mean_psi_z3[:,3] = psi_ave_4keV_z3[:]
mean_psi_z3[:,4] = psi_ave_3keV_z3[:]

mean_psi_z4[:,0] =  psi_ave_cdm_z4[:]
mean_psi_z4[:,1] = psi_ave_9keV_z4[:]
mean_psi_z4[:,2] = psi_ave_6keV_z4[:]
mean_psi_z4[:,3] = psi_ave_4keV_z4[:]
mean_psi_z4[:,4] = psi_ave_3keV_z4[:]

mean_psi_z5[:,0] =  psi_ave_cdm_z5[:]
mean_psi_z5[:,1] = psi_ave_9keV_z5[:]
mean_psi_z5[:,2] = psi_ave_6keV_z5[:]
mean_psi_z5[:,3] = psi_ave_4keV_z5[:]
mean_psi_z5[:,4] = psi_ave_3keV_z5[:]

error_psi_z1[:,0] =  psi_error_cdm_z1[:]
error_psi_z1[:,1] = psi_error_9keV_z1[:]
error_psi_z1[:,2] = psi_error_6keV_z1[:]
error_psi_z1[:,3] = psi_error_4keV_z1[:]
error_psi_z1[:,4] = psi_error_3keV_z1[:]

error_psi_z2[:,0] =  psi_error_cdm_z2[:]
error_psi_z2[:,1] = psi_error_9keV_z2[:]
error_psi_z2[:,2] = psi_error_6keV_z2[:]
error_psi_z2[:,3] = psi_error_4keV_z2[:]
error_psi_z2[:,4] = psi_error_3keV_z2[:]

error_psi_z3[:,0] =  psi_error_cdm_z3[:]
error_psi_z3[:,1] = psi_error_9keV_z3[:]
error_psi_z3[:,2] = psi_error_6keV_z3[:]
error_psi_z3[:,3] = psi_error_4keV_z3[:]
error_psi_z3[:,4] = psi_error_3keV_z3[:]

error_psi_z4[:,0] =  psi_error_cdm_z4[:]
error_psi_z4[:,1] = psi_error_9keV_z4[:]
error_psi_z4[:,2] = psi_error_6keV_z4[:]
error_psi_z4[:,3] = psi_error_4keV_z4[:]
error_psi_z4[:,4] = psi_error_3keV_z4[:]

error_psi_z5[:,0] =  psi_error_cdm_z5[:]
error_psi_z5[:,1] = psi_error_9keV_z5[:]
error_psi_z5[:,2] = psi_error_6keV_z5[:]
error_psi_z5[:,3] = psi_error_4keV_z5[:]
error_psi_z5[:,4] = psi_error_3keV_z5[:]
# time to plot then

fig, axs = plt.subplots(1, 5,figsize=(25,5))
for i in range(n):
    axs[0].fill_between(z_re, mean_psi_z1[:,i] - error_psi_z1[:,i], mean_psi_z1[:,i] + error_psi_z1[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[1].fill_between(z_re, mean_psi_z2[:,i] - error_psi_z2[:,i], mean_psi_z2[:,i] + error_psi_z2[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[2].fill_between(z_re, mean_psi_z3[:,i] - error_psi_z3[:,i], mean_psi_z3[:,i] + error_psi_z3[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[3].fill_between(z_re, mean_psi_z4[:,i] - error_psi_z4[:,i], mean_psi_z4[:,i] + error_psi_z4[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
    axs[4].fill_between(z_re, mean_psi_z5[:,i] - error_psi_z5[:,i], mean_psi_z5[:,i] + error_psi_z5[:,i], facecolor=colors[i], edgecolor=colors[i], label=labels[i])
# axs[0].plot(z_re, psi_ave_cdm_z1, '-', color='royalblue', label=r'cdm')
# axs[0].plot(z_re, psi_ave_3keV_z1, ':', color='magenta', label=r'$m_{\rm wdm} = 3$ keV')
# axs[0].plot(z_re, psi_ave_4keV_z1, ':', color='orange', label=r'$m_{\rm wdm} = 4$ keV')
# axs[0].plot(z_re, psi_ave_6keV_z1, '--', color='purple', label=r'$m_{\rm wdm} = 6$ keV')
# axs[0].plot(z_re, psi_ave_9keV_z1, '-.', color='green', label=r'$m_{\rm wdm} = 9$ keV')
# # axs[0].fill_between(z_re, psi_ave_cdm_z1 - psi_error_cdm_z1, psi_ave_cdm_z1 + psi_error_cdm_z1, alpha=0.2, facecolor='royalblue')
# # axs[0].fill_between(z_re, psi_ave_3keV_z1 - psi_error_3keV_z1, psi_ave_3keV_z1 + psi_error_3keV_z1, alpha=0.2, facecolor='magenta')
# # axs[0].fill_between(z_re, psi_ave_4keV_z1 - psi_error_4keV_z1, psi_ave_4keV_z1 + psi_error_4keV_z1, alpha=0.2, facecolor='orange')
# # axs[0].fill_between(z_re, psi_ave_6keV_z1 - psi_error_6keV_z1, psi_ave_6keV_z1 + psi_error_6keV_z1, alpha=0.2, facecolor='purple')
# # axs[0].fill_between(z_re, psi_ave_9keV_z1 - psi_error_9keV_z1, psi_ave_9keV_z1 + psi_error_9keV_z1, alpha=0.2, facecolor='green')
axs[0].grid(linestyle='dotted')
axs[0].set_title(r'$z_{\rm obs} = 2.0$', fontsize=16)
axs[0].set_xlabel(r'$z_{\rm re}$',fontsize=16)
axs[0].set_ylabel(r'Transparency    [%]',fontsize=16)
axs[0].legend(loc='best', fontsize=14)
axs[0].tick_params(axis = 'both', which = 'major', labelsize = 14)
# next
axs[1].grid(linestyle='dotted')
axs[1].set_title(r'$z_{\rm obs} = 2.5$', fontsize=16)
axs[1].set_xlabel(r'$z_{\rm re}$',fontsize=16)
axs[1].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[2].grid(linestyle='dotted')
axs[2].set_title(r'$z_{\rm obs} = 3.0$', fontsize=16)
axs[2].set_xlabel(r'$z_{\rm re}$',fontsize=16)
axs[2].tick_params(axis = 'both', which = 'major', labelsize = 14)
axs[3].grid(linestyle='dotted')
axs[3].set_title(r'$z_{\rm obs} = 3.5$', fontsize=16)
axs[3].set_xlabel(r'$z_{\rm re}$',fontsize=16)
axs[3].tick_params(axis = 'both', which = 'major', labelsize = 14)
# next
# axs[1].plot(z_re, psi_ave_cdm_z5, '-', color='royalblue', label=r'cdm')
# axs[1].plot(z_re, psi_ave_3keV_z5, ':', color='magenta', label=r'$m_{\rm wdm} = 3$ keV')
# axs[1].plot(z_re, psi_ave_4keV_z5, ':', color='orange', label=r'$m_{\rm wdm} = 4$ keV')
# axs[1].plot(z_re, psi_ave_6keV_z5, '--', color='purple', label=r'$m_{\rm wdm} = 6$ keV')
# axs[1].plot(z_re, psi_ave_9keV_z5, '-.', color='green', label=r'$m_{\rm wdm} = 9$ keV')
# axs[1].fill_between(z_re, psi_ave_cdm_z5 - psi_error_cdm_z5, psi_ave_cdm_z5 + psi_error_cdm_z5, alpha=0.2, facecolor='royalblue')
# axs[1].fill_between(z_re, psi_ave_3keV_z5 - psi_error_3keV_z5, psi_ave_3keV_z5 + psi_error_3keV_z5, alpha=0.2, facecolor='magenta')
# axs[1].fill_between(z_re, psi_ave_4keV_z5 - psi_error_4keV_z5, psi_ave_4keV_z5 + psi_error_4keV_z5, alpha=0.2, facecolor='orange')
# axs[1].fill_between(z_re, psi_ave_6keV_z5 - psi_error_6keV_z5, psi_ave_6keV_z5 + psi_error_6keV_z5, alpha=0.2, facecolor='purple')
# axs[1].fill_between(z_re, psi_ave_9keV_z5 - psi_error_9keV_z5, psi_ave_9keV_z5 + psi_error_9keV_z5, alpha=0.2, facecolor='green')
axs[4].grid(linestyle='dotted')
axs[4].set_title(r'$z_{\rm obs} = 4.0$', fontsize=16)
axs[4].set_xlabel(r'$z_{\rm re}$',fontsize=16)
# axs[1].set_ylabel(r'Transparency    [%]',fontsize=16)
# axs[1].legend(loc='best', fontsize=14)
axs[4].tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.savefig('transparency.pdf',bbox_inches="tight")
plt.show()

# probably fill between better for this. 

 