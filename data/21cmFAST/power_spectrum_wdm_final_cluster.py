import h5py
import numpy as np
import os
import pandas as pd
import cmath
import numpy.fft


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

path = "/work/catalinam/wdmm/wdmm6"
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
#lista_3 = []
for z in redshift_list:
    print('inicial dentro del loop',in_bin_ct)
    in_bin_ct = np.zeros(int(NUM_BINS+2))
    print('despues de definirlo', in_bin_ct)
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
                    print(reading_file.keys())
                    name_delta_xh1 = name_delta_xh.replace("<KeysViewHDF5 ['density'", "") #there are needed to open the data
                    xh_box_name1 = name_delta_xh1.replace(", '", "")
                    xh_box_name = xh_box_name1.replace("']>", "")
                    print(xh_box_name)
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
                    print(range(HII_DIM_one))
                    print(range(HII_DIM))
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
                                        if ct == 18:
                                            print(ct, in_bin_ct[ct])
                                        print(n_x,n_y,n_z)
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


file_new = pd.DataFrame({ 'k':lista_1})
file_new2 = pd.DataFrame({ 'Pm':lista_2})
file_new3 = pd.DataFrame({'z':final_z_list})
#file_new4 = pd.DataFrame({'z':lista_3})

new = file_new.style.hide_index()
file_new.to_csv('results_k_m6',index=False)
new2 = file_new2.style.hide_index()
file_new2.to_csv('results_pm_m6',index=False)
new3 = file_new3.style.hide_index()
file_new3.to_csv('redshift_m6',index=False)
#new2 = file_new4.style.hide_index()
#file_new4.to_csv('results_pm_9_nodiv',index=False)
file_new4 = pd.DataFrame({'z':final_z_list, 'k':lista_1, 'Pm':lista_2 })
new4= file_new4.style.hide_index()
file_new4.to_csv('zpk_data_m6',index=False)
