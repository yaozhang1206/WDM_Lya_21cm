import numpy as np
import pickle
from scipy.interpolate import interp2d

"""
    Build the transparency for lya forest and we want to pickle this function to save time!
    
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
        igm_table = np.loadtxt('../data/gadget/transp_gadget_'+realization+'_'+model+'_f.txt', usecols=[0,1,2,3,4,5,6])
        z_res = [6., 7., 8., 9., 10., 11., 12.]
        z_obs= [2., 2.5, 3., 3.5, 4.]
        self.psi = interp2d(z_res,z_obs, igm_table, kind='cubic')
        
        # the pickle
        file = open('../pickles/psi_'+realization+'_'+model+'.pkl', 'wb')
        pickle.dump(self.psi, file)
        file.close()
