import os
# just a script to run the extrapolation code
# unfortunately I already edited the other one so instead of iterating files which would have been faster, I did this

# besides, it turns out I have to massage the files first due to input
model = ['avg_cdm', 'avg_wdm3', 'avg_wdm4', 'avg_wdm6', 'avg_wdm9', 'r1_cdm', 'r2_cdm', 'r3_cdm', 'r4_cdm', 'r1_wdm3', 'r2_wdm3', 'r3_wdm3', 'r4_wdm3', 'r1_wdm4', 'r2_wdm4', 'r3_wdm4', 'r4_wdm4', 'r1_wdm6', 'r2_wdm6', 'r3_wdm6', 'r4_wdm6', 'r1_wdm9', 'r2_wdm9', 'r3_wdm9', 'r4_wdm9']

def do_the_thing(model):
    file_ini = open('./data/21cmFAST/cross_21cmfast_'+str(model)+'.txt','rt')
    file_out = open('./data/21cmFAST/cross_21cm_'+str(model)+'.txt','wt')
    for line in file_ini:
        file_out.write(line.replace(',',' '))
    file_ini.close()
    file_out.close()

for i in range(0, len(model)):
    do_the_thing(model[i])
