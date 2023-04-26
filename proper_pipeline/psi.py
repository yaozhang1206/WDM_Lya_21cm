import numpy as np
from scipy import interpolate
import pickle
from scipy.interpolate import CloughTocher2DInterpolator

"""
    Build the transparency and we want to pickle this function to save time!
    
    Only needs to be run once per model!
    
"""


class transparency(object):
    
    def __init__(self, params):
        # lya and 21cm needs this, so it will be good to separate it and build it once
        model = params['gadget-model'] # know which file is getting called from gadget
        realization = params['gadget-realization'] # know which realization, or if it is the average one
        # for average it should be avg
        # model may not cover anything for now but we are building up for later
        # grab data
        igm_table = np.loadtxt('../data/gadget/transp_gadget_'+realization+'_'+model+'.txt', usecols=[0,1,2,3,4,5,6])
#        b_gamma = np.loadtxt('./data/gadget/transp_gadget_'+realization+'_'+model+'.txt', usecols=[7]) # do this one somewhere else.
        # for redshift of reionization we have
#        z_res = [6., 7., 8., 9., 10., 11., 12.] # go back after the new sims
        z_res = [6., 7., 8., 9., 10., 11., 12., 6., 7., 8., 9., 10., 11., 12., 6., 7., 8., 9., 10., 11., 12., 6., 7., 8., 9., 10., 11., 12., 6., 7., 8., 9., 10., 11., 12.]
        # need a range of redshift of observations
#        z_obs = [2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5] # go back to this one after new sims
        z_obs = [2., 2., 2., 2., 2., 2., 2., 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3., 3., 3., 3., 3., 3., 3., 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4., 4., 4., 4., 4., 4., 4.]
        redshifts = np.zeros((len(z_obs),2))
        redshifts[:,0] = z_res
        redshifts[:,1] = z_obs
        self.psi = CloughTocher2DInterpolator(redshifts, igm_table.flatten())
        # the pickle
        file = open('../pickles/psi_'+realization+'_'+model+'.pkl', 'wb')
        pickle.dump(self.psi, file)
        file.close()
