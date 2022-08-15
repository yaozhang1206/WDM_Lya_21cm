import py21cmfast as p21c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py21cmfast import plotting
from py21cmfast import cache_tools
from py21cmfast import global_params
import h5py
import cmath
import numpy.fft
import os
import glob, os


#Just to make sure its in the right version
print(f"Using 21cmFAST version {p21c.__version__}")

#Here I produce the neutral hydrogen and density boxes, using the parameters needed
#Global parameters always have to be stated this way, the other ones(cosmo,astro,user) go below run_lightone
with global_params.use(P_CUTOFF= True, M_WDM=3.0 , g_x=1.5,Pop2_ion=4800,OMn=0.0, OMk =0.0, OMr=8.6e-5 , OMtot=1, Y_He=0.245, wl=-1.0, SMOOTH_EVOLVED_DENSITY_FIELD =1, R_smooth_density=0.2, HII_ROUND_ERR= 1e-5, N_POISSON=-1 , MAX_DVDR=0.2,DELTA_R_FACTOR=1.1, DELTA_R_HII_FACTOR=1.1, OPTIMIZE_MIN_MASS=1e11, SHETH_b=0.15, SHETH_c=0.05, ZPRIME_STEP_FACTOR=1.02 ):
    lightcone = p21c.run_lightcone(
        redshift = 5.0, #minimum redshift, next time I will use 5.0
        max_redshift = 15.0, #this is the max, but you always get the data up to z~35
        lightcone_quantities=("brightness_temp", 'density', 'xH_box'), #always put the brightness_temp one, if not it doesnt works
        global_quantities=("brightness_temp", 'density', 'xH_box'),
        user_params = {"HII_DIM": 256, "BOX_LEN": 400,  "DIM":768, "N_THREADS":16  },
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.81,hlittle =0.68 ,OMm = 0.31, OMb =0.04, POWER_INDEX =0.97 ),
        astro_params = {'R_BUBBLE_MAX':50, 'L_X':40.5},
        flag_options = {"INHOMO_RECO": True, "USE_TS_FLUCT":True, "USE_MASS_DEPENDENT_ZETA":True },





        random_seed=54321,
        direc = '/work/catalinam/wdmm/new_run_14_8_2022/r2' #here it is where i want the boxes to be stored
    )


plotting.lightcone_sliceplot(lightcone, "xH_box")
plt.savefig("xh_box_r2_wdm3.png",dpi=200)


#I want to make a file with the neutral hydrogen average value in terms of redshift
avg = lightcone.global_xH #this gives me the neutral hydrogen values
new_str_xh=[]
new_xh_0=[]
for i in reversed(avg):
    new_str_xh.append(str(i))
for i in new_str_xh:
    new_xh_0.append(np.format_float_scientific(float(i), precision = 6, exp_digits=2))

z= lightcone
.node_redshifts #this gives me the values of redshift that go according to the global_xH
new_str_z=[]
new_z_0=[]
for i in reversed(z):
    new_str_z.append(str(i))
for i in new_str_z:
    new_z_0.append(np.format_float_scientific(float(i), precision = 6, exp_digits=2))


file_new = pd.DataFrame({'Redshift':new_z_0, 'xH_avg_value':new_xh_0})
file_new.to_csv('z_and_xH_values_r2_wdm3_15_8_2022',sep=' ',header=False ,index=False)


for f in glob.glob("BrightnessTemp_*.h5"):
    os.remove(f)
for f in glob.glob("TsBox_*.h5"):
    os.remove(f)
##########################################

path='/work/catalinam/wdmm/new_run_14_8_2022/r2'

for filename in os.listdir(path):
    if filename.startswith("IonizedBox_"):
        #print(filename)
        hf = h5py.File(filename, 'a')
        a = hf.attrs
        z1=float(a['redshift'])
        z = str(a['redshift'])
        z_c = z.replace('.', ',')
        f = hf['IonizedBox']
        xh = f['xH_box']
        box_array = np.array(xh)
        new = h5py.File('xh_den'+z+'.h5', 'w')
        new.create_dataset('xhbox'+z_c , data=box_array)
        #print('xh for z is done',z)
        for filename2 in os.listdir(path):
            if filename2.startswith("PerturbedField_"):
                hf2 = h5py.File(filename2, 'r')
                b = hf2.attrs
                z_p = float(b['redshift'])
                if z_p == z1:
                    data = hf2['PerturbedField']
                    rho = data['density']
                    rho_array= np.array(rho)
                    grp=new.create_group('density')
                    grp.create_dataset('density'+z_c,data=rho_array)
                    #print('density done for z=',z_p)
                    hf.close()
                    hf2.close()
                    new.close()

                else:
                    hf2.close()
############################################
for f in glob.glob("PerturbedField_*.h5"):
    os.remove(f)
for f in glob.glob("IonizedBox_*.h5"):
    os.remove(f)
###########################################
k_factor = 1.4
DELTA_K= np.pi*(2)/400 #where 400 is box_len
HII_DIM=256
DIM=768
k_first_bin_ceil = DELTA_K
k_max = DELTA_K*HII_DIM
NUM_BINS = 0
k_floor = 0
k_ceil = k_first_bin_ceil
HII_MIDDLE = (HII_DIM/2)
while (k_ceil < k_max):
    NUM_BINS+=1
    k_floor=k_ceil
    k_ceil*=k_factor
print(NUM_BINS)
p_box = np.zeros(NUM_BINS)
k_ave = np.zeros(NUM_BINS)
in_bin_ct = np.zeros(int(NUM_BINS+2))

path = "/work/catalinam/wdmm/new_run_14_8_2022/r2"
#path_local = '/Users/catalinamorales/Documents/pasantia/wdmm/lightconesf/delta_xh_and_density'

redshift_check = []
for file_1 in os.listdir(path):
    if file_1.startswith("xh_den"):
        file_2 = file_1.replace("xh_den","") #these two replace steps are being used to match the redshift with the density box that is going to be open
        file_3 = file_2.replace(".h5","")
        file_4 = file_3.replace(",", ".")
        redshift_check.append(file_4)
    else:
        continue
redshift_list = sorted(redshift_check, key = lambda x:float(x))
#print('sorted',redshift_list)

file_check = []
#file_check_d = []

final_z_list = []
lista_1 = []
lista_2 = []
lista_1_sn=[]
lista_2_sn=[]
lista_z_sn =[]
#lista_3 = []
for z in redshift_list:
    #print('inicial dentro del loop',in_bin_ct)
    in_bin_ct = np.zeros(int(NUM_BINS+2))
    #print('despues de definirlo', in_bin_ct)
    p_box = np.zeros(NUM_BINS)
    k_ave = np.zeros(NUM_BINS)

    for file_delta_xh in os.listdir(path):
        if file_delta_xh.startswith("xh_den"):
            file_delta_xh1 = file_delta_xh.replace("xh_den","") #these two replace steps are being used to match the redshift with the density box that is going to be open
            file_delta_xh2 = file_delta_xh1.replace(".h5","")
            file_delta_xh3 = file_delta_xh2.replace(",", ".")
            if z == file_delta_xh3:
                #print(z, 'z and delta z', file_delta_xh3)
                if file_delta_xh not in file_check:
                    file_check.append(file_delta_xh)
                    reading_file = h5py.File(file_delta_xh, 'r') #here I read the .h5 file
                    name_delta_xh = str(reading_file.keys())
                    #print(reading_file.keys())
                    name_delta_xh1 = name_delta_xh.replace("<KeysViewHDF5 ['density'", "") #there are needed to open the data
                    xh_box_name1 = name_delta_xh1.replace(", '", "")
                    xh_box_name = xh_box_name1.replace("']>", "")
                    #print(xh_box_name)
                    xh_box = reading_file[xh_box_name]
                    xh_box_array = np.array(xh_box)

                    fft_array = numpy.fft.fftn(xh_box_array, norm='backward') #Applying the real fft to a 3d array
                    #print(fft_array.size, fft_array.shape,'xhbox size', xh_box_array.shape)

                    density_box = reading_file.get('density')
                    density_name= str(density_box.keys())
                    density_name1 = density_name.replace("<KeysViewHDF5 ['", "")
                    density_name2 = density_name1.replace("']>", "")
                    density_data = density_box[density_name2]
                    density_data_array = np.array(density_data)

                    fft_array_den = numpy.fft.fftn(density_data_array,norm='backward')

                    reading_file.close()
                    n_x=0
                     #starting with the power spectrum, getting the k's
                    HII_DIM_one = int(HII_DIM-1)
                    #print(range(HII_DIM_one))
                    #print(range(HII_DIM))
                    for n_x in range(HII_DIM_one):
                        if (n_x>HII_MIDDLE):
                            k_x =(n_x-HII_DIM) * DELTA_K
                            n_x+=1
                        else:
                            k_x = n_x * DELTA_K
                            n_x+=1

                        n_y=0
                        for n_y in range(HII_DIM_one) :
                            if (n_y>HII_MIDDLE):
                                k_y =(n_y-HII_DIM) * DELTA_K
                                n_y+=1
                            else:
                                k_y = n_y * DELTA_K
                                n_y+=1


                            n_z=0
                            for n_z in range(int(HII_MIDDLE+1)) :
                                k_z = n_z * DELTA_K
                                n_z+=1
                                k_mag = np.sqrt(k_x*k_x + k_y*k_y + k_z*k_z)
                                #print('This is k_mag', k_mag)
                                ct = 0
                                k_floor = 0
                                k_ceil = k_first_bin_ceil

                                while (k_ceil < k_max):
                                    #print('k_ceil < k_max is True', k_ceil)
                                    #print(ct)
                                    if (k_mag>=k_floor) and (k_mag < k_ceil) :
                                        #ct = ct+1
                                        #print('inside the if')
                                        in_bin_ct[ct] = in_bin_ct[ct] +1

                                        xh_final = fft_array[n_x][n_y][n_z]
                                        #print(xh_final)

                                        density_final = fft_array_den[n_x][n_y][n_z]
                                        #print(density_final)
                                        #print('Multiplication 1',density_final.real * xh_final.real)
                                        #print('Multiplication 2',density_final.imag * xh_final.imag )
                                        p_box[ct]+= pow(400/256,6)*(pow(k_mag,3)*((density_final.real * xh_final.real)+(density_final.imag * xh_final.imag ))/ (2.0*np.pi*np.pi*pow(400,3)))
                                        k_ave[ct]+=(k_mag)



                                        #print(p_box)
                                        #print(z_order)


                                        break

                                    ct+=1
                                    k_floor=k_ceil
                                    k_ceil*=k_factor



                else:
                    continue

            else:
                continue

        else:
            continue

    for s in range(NUM_BINS):
        final_z_list.append(z)

        lista_1.append(k_ave[s]/(in_bin_ct[s]))
        lista_2.append(p_box[s]/(in_bin_ct[s]))


        #lista_3.append(p_box[s])
        s+=1



file_new4 = pd.DataFrame({'z':final_z_list, 'k':lista_1, 'pm':lista_2 })
new4= file_new4.style.hide_index()
file_new4_1 = file_new4[file_new4['k']!=0]
file_new4_1.to_csv('cross_21cm_r2_wdm3_15_8_2022',sep=' ',header=False ,index=False)

for f in glob.glob("xh_den*.h5"):
    os.remove(f)

z_pd=list(file_new4_1['z'])
k_pd=list(file_new4_1['k'])
pm_pd=list(file_new4_1['pm'])

new_z=[]
new_k=[]
new_pm=[]

for i in z_pd:
    new_z.append(np.format_float_scientific(float(i), precision = 6, exp_digits=2))
for j in k_pd :
    new_k.append(np.format_float_scientific(float(j), precision = 6, exp_digits=2))
for k in pm_pd:
    new_pm.append(np.format_float_scientific(float(k), precision = 6, exp_digits=2))

file_new5=pd.DataFrame({'z':new_z, 'k':new_k, 'pm':new_pm })
new5= file_new5.style.hide_index()
file_new5.to_csv('cross_21cm_r2_wdm3_15_8_2022_sn',sep=' ',header=False ,index=False)
