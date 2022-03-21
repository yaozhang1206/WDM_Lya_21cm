import h5py
import numpy as np
import os
import pandas as pd

z_list=[]

path='/work/catalinam/wdmm/wdmm6'

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
        print('xh for z is done',z)
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
                    print('density done for z=',z_p)
                    hf.close()
                    hf2.close()
                    new.close()
                    z_list.append(float(z))
                else:
                    hf2.close()

#z_np= np.array(z_list)
#np.savetxt('z_list_fid', z_np)
