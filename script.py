import f_mass
import numpy as np
import sys
import pandas as pd
from scipy import interpolate
import wdm

v_bc=33

scenarios = ['3kev_s8', '3kev_sminus', '3kev_splus', '4kev_s8', '4kev_sminus', '4kev_splus',//
'6kev_s8', '6kev_sminus', '6kev_splus', '9kev_s8', '9kev_sminus', '9kev_splus', 'cdm_s8', 'cdm_sminus', 'cdm_splus']

z_res = ['06','07', '08', '09', '10', '11', '12']

directory = '/work/yaozhang/wdm_Gadget-2/Gadget-2.0.7/Gadget2-src1/'
# WDM directory
# directory = '/scratch/yaozhang/3keV_s8_1_xray/HMZ'+z_re+'/'

##############################################################
# Extract sound speed data
##############################################################

for z_re in z_res:
	path=directory + 'HMZ'+z_re+'/'
	# a=np.loadtxt(path+'/outputs_zre' + str(int(z_re)) + '_z2.txt')[:-15] 
	a_list_file = path+'/outputs_zre' + str(int(z_re)) + '_z35.txt'
	a=np.loadtxt(a_list_file)
	z = 1/a-1
	# the commented two lines below are used to add the last snapshot to the array, which is not included in the output file
	# z_l = 1/a[-1]-1 - 0.1
	# a = np.append(a, 1./(1+z_l))
	c_s=[]
	gamma=[]
	T_0=[]
	logT_0=[]
	for i in range(len(a)):
		print(str(len(a))+' '+str(i))
		f=f_mass.f_scale(path+'snapshot_0'+str(i).zfill(2)+'.extract',a[i], z_re)
		c_s_tmp,gamma_tmp,T_0_tmp,logT_0_tmp=f.cs_extract()
		c_s.append(c_s_tmp)
		gamma.append(gamma_tmp)
		T_0.append(T_0_tmp)
		logT_0.append(logT_0_tmp)
	tb={'a':a,'z':z,'c_s':c_s,'gamma':gamma,'T_0':T_0,'logT_0':logT_0}
	# output file name, you could custom these names
	cs_filename='/home/hylong/fiducial/cs_' + z_re + '.csv'
	pd.DataFrame(tb).to_csv(cs_filename)

##############################################################
# After extracting sound speed, calculate filtering masses
##############################################################

for z_re in z_res:
	MF=[]
	kF=[]
	kF_inverse2=[]
	kF_v_inverse2=[]
	kF_s_inverse2=[]

	# cs_filename='/home/hylong/fiducial/cs_' + z_re + '.csv'
	# path = '/scratch/yaozhang/3keV_s8_1_xray/HMZ'+z_re+'/'
	cs_filename='/home/hylong/fiducial/cs_' + z_re + '.csv'
	path= directory + 'HMZ'+z_re+'/'
	tb = pd.read_csv(cs_filename)

	# Be careful about the c_s data selected to use here as sometimes depending on your
	#simulated data the a or z could jump, so make sure they are consistent when analyzing data
	c_s = tb['c_s'].to_numpy()
	# c_s = np.append(c_s[:19], c_s[19::5])
	a = tb['a'].to_numpy()
	# a = np.append(a[:19],a[19::5])
	z = 1/a - 1
	# Interpolate sound speed with respect to a in order to use it as a function
	cs_func=interpolate.interp1d(np.array(a),np.array(c_s), fill_value="extrapolate")
	for i in np.arange(len(a)):
		f=f_mass.f_scale(path+'/snapshot_0'+str(i).zfill(2)+'.extract',a[i], z_re)
		MF_tmp,kF_v_inverse2_tmp,kF_s_inverse2_tmp,kF_tmp,kF_inverse2_tmp = f.results(cs_func,v_bc)
		MF.append(MF_tmp)
		kF_v_inverse2.append(kF_v_inverse2_tmp)
		kF_s_inverse2.append(kF_s_inverse2_tmp)
		kF.append(kF_tmp)
		kF_inverse2.append(kF_inverse2_tmp)
		print('At redshift z='+str(z[i])+', a='+str(a[i]))
		print('Filtering mass M_F='+str(MF[-1])+' , filtering scale k_F='+str(kF[-1])+', k_F^-2='+str(kF_inverse2[-1])+' ,k_Fv^-2='+str(kF_v_inverse2[-1])+' ,k_Fs^-2='+str(kF_s_inverse2[-1]))
	data={'a':a,'z':z,'MF':MF,'kF':kF,'kF_inverse2':kF_inverse2,'kF_v_inverse2':kF_v_inverse2,'kF_s_inverse2':kF_s_inverse2}
	# Again, you should custom the output directory and names
	filename='/home/hylong/fiducial/f_scale_'+z_re+'.csv'

	# output filtering mass data to a csv file
	print('data --> '+ filename+'\n')
	pd.DataFrame(data).to_csv(filename)


# For the sake of analysis convenience, I reorganized the filtering masses data in multiple files and
# save the data in a single cvs file
zres=['06','07','08','09','10','11','12']

mf=[]
mf_std=[]
for z_re in zres:
	# change the directory to the one where you saved the file
    dt1 = pd.read_csv('./fiducial/f_scale_'+z_re+'.csv')
    # [:-21] below means go from z=5.5 to z=3.5, this should change when you have 
    # different form of a or z list from the last steps 
    mf1=dt1['MF'].to_numpy()[-21:]
    mf.append(mf1)

ind = [5.5, 5.4, 5.3, 5.2, 5.1, 5.0 , 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0 , 3.9, 3.8, 3.7, 3.6,3.5]
# ind = [5.5, 5.0, 4.5, 4.0, 3.5]
# save them to a new dataframe
df_mean=pd.DataFrame(np.array(mf).T,columns=['6.0','7.0','8.0','9.0','10.0','11.0','12.0'],index=ind)
df_mean.to_csv('./fiducial/fmass.csv',index_label='z_obs')



############################################################################
# Now calculating rho_HI and make interpolated funcitons and save them
############################################################################
import wdm
from scipy import interpolate

######################################
# Don't worry about these lines, I just want to use some functions inside of wdm.P_patchy_reion
# so I need to initiate an instance

k = 0.1
mu = 0.1
z = 3.5
params = {}
params['h'] = 0.6774
params['Obh2'] = 0.02230
params['Och2'] = 0.1188
params['mnu'] = 0.194
params['As'] = 2.142e-9
params['ns'] = 0.9667
params['alphas'] = -0.002
params['taure'] = 0.066
params['O_HI'] = 1.18e-3
params['bHI'] = 2.3970

reion_his_file = './xH_r1_wdm3.txt'
cross_ps_file = 'cross_21cm_r1_wdm3.txt'

rho_calc = wdm.P_patchy_reion(k, mu, z, params, reion_his_file, cross_ps_file)
rho_calc.halo_func()
##############################################################

# Calculating rho_HI
f_b = 0.1573
z_res=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
rho_HI = []
coords = []

df_mean = pd.read_csv('./fiducial/fmass.csv',index_label='z_obs')
MF = df_mean
for i in range(len(z_res)):
    for zobs in MF.index.to_numpy():
    	coords.append([z_res[i],zobs])
        M_baryon = np.array([f_b*Mhalo*(1+(2**(1./3)-1)*MF.loc[zobs][i]/Mhalo)**(-3.) for Mhalo in rho_calc.Mhalos])
        rho_HI.append(integrate.simps(rho_calc.dndM*M_baryon_large, rho_calc.Mhalos))

rho_HI = np.array(rho_HI).reshape(len(z_res),-1)

# get rho_func
rho_func = interpolate.LinearNDInterpolator(coords, np.array(rho_HI).flatten())

# save function in pickle file
import pickle

with open('./rho_HI_func.pkl','wb') as f:
	pickle.dump(rho_func, f)







