import py21cmfast as p21c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py21cmfast import plotting
from py21cmfast import cache_tools
from py21cmfast import global_params
from powerbox import PowerBox
from powerbox.tools import get_power
import h5py
import os
#Just to make sure its in the right version
print(f"Using 21cmFAST version {p21c.__version__}")

#Here I produce the neutral hydrogen and density boxes, using the parameters needed
#Global parameters always have to be stated this way, the other ones(cosmo,astro,user) go below run_lightone
with global_params.use(P_CUTOFF= True, M_WDM=6.0 , g_x=1.5,Pop2_ion=4800,OMn=0.0, OMk =0.0, OMr=8.6e-5 , OMtot=1, Y_He=0.245, wl=-1.0, SMOOTH_EVOLVED_DENSITY_FIELD =1, R_smooth_density=0.2, HII_ROUND_ERR= 1e-5, N_POISSON=-1 , MAX_DVDR=0.2,DELTA_R_FACTOR=1.1, DELTA_R_HII_FACTOR=1.1, OPTIMIZE_MIN_MASS=1e11, SHETH_b=0.15, SHETH_c=0.05, ZPRIME_STEP_FACTOR=1.02 ):
    lightcone = p21c.run_lightcone(
        redshift = 5.0, #minimum redshift, next time I will use 5.0
        max_redshift = 15.0, #this is the max, but you always get the data up to z~35
        lightcone_quantities=("brightness_temp", 'density', 'xH_box'), #always put the brightness_temp one, if not it doesnt works
        global_quantities=("brightness_temp", 'density', 'xH_box'),
        user_params = {"HII_DIM": 256, "BOX_LEN": 400,  "DIM":768, "N_THREADS":16  },
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.81,hlittle =0.68 ,OMm = 0.31, OMb =0.04, POWER_INDEX =0.97 ),
        astro_params = {'R_BUBBLE_MAX':50, 'L_X':40.5},
        flag_options = {"INHOMO_RECO": True, "USE_TS_FLUCT":True, "USE_MASS_DEPENDENT_ZETA":True },





        random_seed=12345,
        direc = '/work/catalinam/wdmm/wdmm6' #here it is where i want the boxes to be stored
    )

avg = lightcone.global_xH #this gives me the neutral hydrogen values
avg1 = np.array(avg)
np.savetxt('xh_list', avg1)
z= lightcone.node_redshifts #this gives me the values of redshift that go according to the global_xH
z1 = np.array(z)
np.savetxt('z_list', z1)
pd_file= pd.DataFrame({'z':z1, 'avg':avg1 })
new4= pd_file.style.hide_index()
pd_file.to_csv('xh_avg_val_6.txt',index=False)
