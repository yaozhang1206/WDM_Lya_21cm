import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import pylab



"""This code generates the global history reionization"""
# first define the tanh model
# but first the y function
def useful_y(z):
    """ used in the hyperbolic tangent model"""
    return 1.0 * pow(1.0 + z, 1.5)
def tanH_model(z_re,z):
    """ hyperbolyc tangent model, without Helium contribution """
    delta_z = 0.5 # as defined by Planck
    y_z_re = useful_y(z_re)
    y_z = useful_y(z)
    delta_y = 1.5 * pow(1.0 + z_re, 0.5) * delta_z # this is definitely a derivative
    temp = y_z_re - y_z
    temp = temp / (1.0 * delta_y)
    # now we will include the helium correction
    Y_factor = 0.2454
    He_ratio = Y_factor / (1.0 - Y_factor) / 4.0
    He_ratio = 0
    x_e = (1.0 + He_ratio) * (1.0 + np.tanh(temp)) / 2.0
    return x_e
# let's grab all the data
# grab the wdm candidates
#z, xH_3keV = pylab.loadtxt('./data/21cmFAST/xH_test_3keV.txt',unpack=True)
#z, xH_4keV = pylab.loadtxt('./data/21cmFAST/xH_test_4keV.txt',unpack=True)
#z, xH_6keV = pylab.loadtxt('./data/21cmFAST/xH_test_6keV.txt',unpack=True)
#z, xH_9keV = pylab.loadtxt('./data/21cmFAST/xH_test_9keV.txt',unpack=True)
# grab the fiducial
#z, xH_cdm = pylab.loadtxt('./data/21cmFAST/xH_test_cdm.txt',unpack=True)
# I am going to need to invert/interpolate for future plot
#history_cdm = interp1d(xH_cdm, z)
#history_3keV = interp1d(xH_3keV, z)
#history_4keV = interp1d(xH_4keV, z)
#history_6keV = interp1d(xH_6keV, z)
#history_9keV = interp1d(xH_9keV, z)
# now I have to actually grab Catalina's data of the different realizations
# start with cdm
z, xH_r1_cdm = np.loadtxt('../data/21cmFAST/xH_21cm_r1_cdm_s8.txt', unpack=True)
z, xH_r2_cdm = np.loadtxt('../data/21cmFAST/xH_21cm_r2_cdm_s8.txt', unpack=True)
z, xH_r3_cdm = np.loadtxt('../data/21cmFAST/xH_21cm_r3_cdm_s8.txt', unpack=True)
z, xH_r4_cdm = np.loadtxt('../data/21cmFAST/xH_21cm_r4_cdm_s8.txt', unpack=True)
# 3keV
z, xH_r1_3keV = np.loadtxt('../data/21cmFAST/xH_21cm_r1_3keV_s8.txt', unpack=True)
z, xH_r2_3keV = np.loadtxt('../data/21cmFAST/xH_21cm_r2_3keV_s8.txt', unpack=True)
z, xH_r3_3keV = np.loadtxt('../data/21cmFAST/xH_21cm_r3_3keV_s8.txt', unpack=True)
z, xH_r4_3keV = np.loadtxt('../data/21cmFAST/xH_21cm_r4_3keV_s8.txt', unpack=True)
# 4keV
z, xH_r1_4keV = np.loadtxt('../data/21cmFAST/xH_21cm_r1_4keV_s8.txt', unpack=True)
z, xH_r2_4keV = np.loadtxt('../data/21cmFAST/xH_21cm_r2_4keV_s8.txt', unpack=True)
z, xH_r3_4keV = np.loadtxt('../data/21cmFAST/xH_21cm_r3_4keV_s8.txt', unpack=True)
z, xH_r4_4keV = np.loadtxt('../data/21cmFAST/xH_21cm_r4_4keV_s8.txt', unpack=True)
# 6keV
z, xH_r1_6keV = np.loadtxt('../data/21cmFAST/xH_21cm_r1_6keV_s8.txt', unpack=True)
z, xH_r2_6keV = np.loadtxt('../data/21cmFAST/xH_21cm_r2_6keV_s8.txt', unpack=True)
z, xH_r3_6keV = np.loadtxt('../data/21cmFAST/xH_21cm_r3_6keV_s8.txt', unpack=True)
z, xH_r4_6keV = np.loadtxt('../data/21cmFAST/xH_21cm_r4_6keV_s8.txt', unpack=True)
# 9keV
z, xH_r1_9keV = np.loadtxt('../data/21cmFAST/xH_21cm_r1_9keV_s8.txt', unpack=True)
z, xH_r2_9keV = np.loadtxt('../data/21cmFAST/xH_21cm_r2_9keV_s8.txt', unpack=True)
z, xH_r3_9keV = np.loadtxt('../data/21cmFAST/xH_21cm_r3_9keV_s8.txt', unpack=True)
z, xH_r4_9keV = np.loadtxt('../data/21cmFAST/xH_21cm_r4_9keV_s8.txt', unpack=True)
# now let's compute the average and standard deviation on the mean
def ave(a,b,c,d):
    return (a + b + c + d)/4.

def std(a,b,c,d):
    mean = ave(a,b,c,d)
    s = (a - mean)**2 + (b - mean)**2 + (c - mean)**2 + (d - mean)**2
    s = s / 4.
    return np.sqrt(s)
    
mean_cdm  = ave(xH_r1_cdm, xH_r2_cdm, xH_r3_cdm, xH_r4_cdm)
error_cdm = std(xH_r1_cdm, xH_r2_cdm, xH_r3_cdm, xH_r4_cdm)
mean_3keV  = ave(xH_r1_3keV, xH_r2_3keV, xH_r3_3keV, xH_r4_3keV)
error_3keV = std(xH_r1_3keV, xH_r2_3keV, xH_r3_3keV, xH_r4_3keV)
mean_4keV  = ave(xH_r1_4keV, xH_r2_4keV, xH_r3_4keV, xH_r4_4keV)
error_4keV = std(xH_r1_4keV, xH_r2_4keV, xH_r3_4keV, xH_r4_4keV)
mean_6keV  = ave(xH_r1_6keV, xH_r2_6keV, xH_r3_6keV, xH_r4_6keV)
error_6keV = std(xH_r1_6keV, xH_r2_6keV, xH_r3_6keV, xH_r4_6keV)
mean_9keV  = ave(xH_r1_9keV, xH_r2_9keV, xH_r3_9keV, xH_r4_9keV)
error_9keV = std(xH_r1_9keV, xH_r2_9keV, xH_r3_9keV, xH_r4_9keV)

# I am going to need to invert/interpolate for future plot
history_cdm = interp1d(mean_cdm, z)
history_3keV = interp1d(mean_3keV, z)
history_4keV = interp1d(mean_4keV, z)
history_6keV = interp1d(mean_6keV, z)
history_9keV = interp1d(mean_9keV, z)

n = 5
colors = plt.cm.viridis(np.linspace(0,1,n))
labels = [r'CDM', r'm$_{\rm wdm} = 9$ keV', r'm$_{\rm wdm} = 6$ keV', r'm$_{\rm wdm} = 4$ keV', r'm$_{\rm wdm} = 3$ keV']

xH_table = np.zeros((len(z),5))
xH_table[:,0] = mean_cdm[:]
xH_table[:,1] = mean_9keV[:]
xH_table[:,2] = mean_6keV[:]
xH_table[:,3] = mean_4keV[:]
xH_table[:,4] = mean_3keV[:]
xH_e_table = np.zeros((len(z),5))
xH_e_table[:,0] = error_cdm[:]
xH_e_table[:,1] = error_9keV[:]
xH_e_table[:,2] = error_6keV[:]
xH_e_table[:,3] = error_4keV[:]
xH_e_table[:,4] = error_3keV[:]


# and then we plot them
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
#ax1.plot(z,mean_cdm,color='black',zorder=4)
#ax1.plot(z,mean_3keV,color='purple',zorder=4)
#ax1.plot(z,mean_4keV,color='salmon',zorder=4)
#ax1.plot(z,mean_6keV,color='aqua',zorder=4)
#ax1.plot(z,mean_9keV,color='chartreuse',zorder=4)
ax1.axvline(x=5.90,linestyle='dashed',color='navy')
for i in range(n):
    ax1.fill_between(z, xH_table[:,i] - xH_e_table[:,i], xH_table[:,i] + xH_e_table[:,i], facecolor = colors[i], edgecolor=colors[i], label=labels[i])


#ax1.fill_between(z, mean_cdm - error_cdm, mean_cdm + error_cdm, facecolor='black', edgecolor='black', zorder=3, alpha=0.8, label=r'CDM')
#ax1.fill_between(z, mean_3keV - error_3keV, mean_3keV + error_3keV, facecolor='purple', edgecolor='purple', zorder=3, alpha=0.8, label= r'm$_{\rm wdm} = 3$ keV')
#ax1.fill_between(z, mean_4keV - error_4keV, mean_4keV + error_4keV, facecolor='salmon', edgecolor='salmon', zorder=3, alpha=0.8, label=r'm$_{\rm wdm} = 4$ keV')
#ax1.fill_between(z, mean_6keV - error_6keV, mean_6keV + error_6keV, facecolor='aqua', edgecolor='aqua', zorder=3, alpha=0.8,label=r'm$_{\rm wdm} = 6$ keV')
#ax1.fill_between(z, mean_9keV - error_9keV, mean_9keV + error_9keV, facecolor='chartreuse', edgecolor='chartreuse', zorder=3, alpha=0.8,label=r'm$_{\rm wdm} = 9$ keV')
ax1.plot(z,(1.0 - tanH_model(7.68 - 0.79,z)),':',color='maroon',label=r'Planck 1-$\sigma$',zorder=5)
ax1.plot(z,(1.0 - tanH_model(7.68 + 0.79,z)),':',color='maroon',zorder=5)

ax1.errorbar(6.3,0.79,0.04,uplims=True,marker='o', color='gold',markersize=7,label=r'Dark Pixel 2015',zorder=6)
ax1.errorbar(6.5,0.87,0.03,uplims=True,marker='o', color='gold',markersize=7,zorder=6)
ax1.errorbar(6.7,0.94,0.09,uplims=True,marker='o', color='gold',markersize=7,zorder=6)
ax1.errorbar(6.1,0.69,0.06,uplims=True,marker='o', color='gold',markersize=7,zorder=6)
ax1.errorbar(5.9,0.11,0.06,uplims=True,marker='o',markersize=7,label=r'Dark Pixel 2023',zorder=6)
ax1.errorbar(6.9,0.4,0.1,uplims=False,lolims=True,marker='s',markersize=7,label=r'Ly$\alpha$ Fraction',zorder=6)
ax1.errorbar(6.6,0.5,0.1,uplims=True,lolims=False,marker='*',markersize=7,label=r'LAE Clustering',zorder=6)
ax1.errorbar(7.,0.7,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J0252-0503',zorder=6)
ax1.errorbar(7.5,0.39,yerr=np.array([[0.13,0.22]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1007+2115',zorder=6)
ax1.errorbar(7.1,0.4,yerr=np.array([[0.19,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1120+0641',zorder=6)
ax1.errorbar(7.5,0.21,yerr=np.array([[0.19,0.17]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928a',zorder=6)
ax1.errorbar(7.6,0.56,yerr=np.array([[0.18,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928b',zorder=6)
ax1.errorbar(7.4,0.60,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928c',zorder=6)
ax1.errorbar(7.29,0.49,yerr=np.array([[0.11,0.11]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'Combined quasars',zorder=6)
ax1.errorbar(7.0,0.59,yerr=np.array([[0.15,0.11]]).T,xerr=0.5,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWa',zorder=6)
ax1.errorbar(7.6,0.88,yerr=np.array([[0.10,0.05]]).T,xerr=0.6,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWb',zorder=6)
ax1.errorbar(8.0,0.76,yerr=0.22,xerr=0.6,uplims=False,lolims=True,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWc',zorder=6)

ax1.set_xlim(5.5,10)
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'Neutral hydrogen fraction',fontsize=14)
#ax1.legend(loc='lower right')
ax1.legend(loc='center left',bbox_to_anchor=(1.,0.5))
#plt.savefig('global_history_yao.pdf',bbox_inches="tight")
plt.savefig('global_history.png',bbox_inches="tight")
plt.show()

# let's also do the mini-halos
# start with cdm
z, mh_xH_r1_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r1_cdm.txt', unpack=True)
z, mh_xH_r2_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r2_cdm.txt', unpack=True)
z, mh_xH_r3_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r3_cdm.txt', unpack=True)
z, mh_xH_r4_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r4_cdm.txt', unpack=True)
# 3keV
z, mh_xH_r1_3keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r1_wdm3.txt', unpack=True)
z, mh_xH_r2_3keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r2_wdm3.txt', unpack=True)
z, mh_xH_r3_3keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r3_wdm3.txt', unpack=True)
z, mh_xH_r4_3keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r4_wdm3.txt', unpack=True)
# 4keV
z, mh_xH_r1_4keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r1_wdm4.txt', unpack=True)
z, mh_xH_r2_4keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r2_wdm4.txt', unpack=True)
z, mh_xH_r3_4keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r3_wdm4.txt', unpack=True)
z, mh_xH_r4_4keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r4_wdm4.txt', unpack=True)
# 6keV
z, mh_xH_r1_6keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r1_wdm6.txt', unpack=True)
z, mh_xH_r2_6keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r2_wdm6.txt', unpack=True)
z, mh_xH_r3_6keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r3_wdm6.txt', unpack=True)
z, mh_xH_r4_6keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r4_wdm6.txt', unpack=True)
# 9keV
z, mh_xH_r1_9keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r1_wdm9.txt', unpack=True)
z, mh_xH_r2_9keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r2_wdm9.txt', unpack=True)
z, mh_xH_r3_9keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r3_wdm9.txt', unpack=True)
z, mh_xH_r4_9keV = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_r4_wdm9.txt', unpack=True)

mh_mean_cdm  = ave(mh_xH_r1_cdm, mh_xH_r2_cdm, mh_xH_r3_cdm, mh_xH_r4_cdm)
mh_error_cdm = std(mh_xH_r1_cdm, mh_xH_r2_cdm, mh_xH_r3_cdm, mh_xH_r4_cdm)
mh_mean_3keV  = ave(mh_xH_r1_3keV, mh_xH_r2_3keV, mh_xH_r3_3keV, mh_xH_r4_3keV)
mh_error_3keV = std(mh_xH_r1_3keV, mh_xH_r2_3keV, mh_xH_r3_3keV, mh_xH_r4_3keV)
mh_mean_4keV  = ave(mh_xH_r1_4keV, mh_xH_r2_4keV, mh_xH_r3_4keV, mh_xH_r4_4keV)
mh_error_4keV = std(mh_xH_r1_4keV, mh_xH_r2_4keV, mh_xH_r3_4keV, mh_xH_r4_4keV)
mh_mean_6keV  = ave(mh_xH_r1_6keV, mh_xH_r2_6keV, mh_xH_r3_6keV, mh_xH_r4_6keV)
mh_error_6keV = std(mh_xH_r1_6keV, mh_xH_r2_6keV, mh_xH_r3_6keV, mh_xH_r4_6keV)
mh_mean_9keV  = ave(mh_xH_r1_9keV, mh_xH_r2_9keV, mh_xH_r3_9keV, mh_xH_r4_9keV)
mh_error_9keV = std(mh_xH_r1_9keV, mh_xH_r2_9keV, mh_xH_r3_9keV, mh_xH_r4_9keV)

# I am going to need to invert/interpolate for future plot
mh_history_cdm = interp1d(mh_mean_cdm, z)
mh_history_3keV = interp1d(mh_mean_3keV, z)
mh_history_4keV = interp1d(mh_mean_4keV, z)
mh_history_6keV = interp1d(mh_mean_6keV, z)
mh_history_9keV = interp1d(mh_mean_9keV, z)

#arr = mean_cdm[::-1]
#arr_error = error_cdm[::-1]

# for reference
znmh, nmh_xH_r1_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_nmh_xh_r1_cdm.txt', unpack=True)
znmh, nmh_xH_r2_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_nmh_r2_cdm.txt', unpack=True)
znmh, nmh_xH_r3_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_nmh_r3_cdm.txt', unpack=True)
znmh, nmh_xH_r4_cdm = np.loadtxt('../data/21cmFAST/mini_halos/21cmfast_xh_nmh_r4_cdm.txt', unpack=True)

nmh_mean_cdm = ave(nmh_xH_r1_cdm, nmh_xH_r2_cdm, nmh_xH_r3_cdm, nmh_xH_r4_cdm)
nmh_error_cdm = std(nmh_xH_r1_cdm, nmh_xH_r2_cdm, nmh_xH_r3_cdm, nmh_xH_r4_cdm)

nmh_history_cdm = interp1d(nmh_mean_cdm, z)

mh_xH_table = np.zeros((len(z),6))
#mh_xH_table[:,0] = arr[:]
mh_xH_table[:,0] = nmh_mean_cdm[:]
mh_xH_table[:,1] = mh_mean_cdm[:]
mh_xH_table[:,2] = mh_mean_9keV[:]
mh_xH_table[:,3] = mh_mean_6keV[:]
mh_xH_table[:,4] = mh_mean_4keV[:]
mh_xH_table[:,5] = mh_mean_3keV[:]
mh_xH_e_table = np.zeros((len(z),6))
#mh_xH_e_table[:,0] = arr_error[:]
mh_xH_e_table[:,0] = nmh_error_cdm[:]
mh_xH_e_table[:,1] = mh_error_cdm[:]
mh_xH_e_table[:,2] = mh_error_9keV[:]
mh_xH_e_table[:,3] = mh_error_6keV[:]
mh_xH_e_table[:,4] = mh_error_4keV[:]
mh_xH_e_table[:,5] = mh_error_3keV[:]



mh_n = 6
mh_colors = plt.cm.viridis(np.linspace(0,1,mh_n))
mh_labels = [r'CDM', r'CDM-MH', r'm$_{\rm wdm} = 9$ keV', r'm$_{\rm wdm} = 6$ keV', r'm$_{\rm wdm} = 4$ keV', r'm$_{\rm wdm} = 3$ keV']

# and then we plot them
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
#ax1.plot(z,mean_cdm,color='black',zorder=4)
#ax1.plot(z,mean_3keV,color='purple',zorder=4)
#ax1.plot(z,mean_4keV,color='salmon',zorder=4)
#ax1.plot(z,mean_6keV,color='aqua',zorder=4)
#ax1.plot(z,mean_9keV,color='chartreuse',zorder=4)
ax1.axvline(x=5.90,linestyle='dashed',color='navy')
for i in range(mh_n):
    ax1.fill_between(z, mh_xH_table[:,i] - mh_xH_e_table[:,i], mh_xH_table[:,i] + mh_xH_e_table[:,i], facecolor = mh_colors[i], edgecolor=mh_colors[i], label=mh_labels[i])


#ax1.fill_between(z, mean_cdm - error_cdm, mean_cdm + error_cdm, facecolor='black', edgecolor='black', zorder=3, alpha=0.8, label=r'CDM')
#ax1.fill_between(z, mean_3keV - error_3keV, mean_3keV + error_3keV, facecolor='purple', edgecolor='purple', zorder=3, alpha=0.8, label= r'm$_{\rm wdm} = 3$ keV')
#ax1.fill_between(z, mean_4keV - error_4keV, mean_4keV + error_4keV, facecolor='salmon', edgecolor='salmon', zorder=3, alpha=0.8, label=r'm$_{\rm wdm} = 4$ keV')
#ax1.fill_between(z, mean_6keV - error_6keV, mean_6keV + error_6keV, facecolor='aqua', edgecolor='aqua', zorder=3, alpha=0.8,label=r'm$_{\rm wdm} = 6$ keV')
#ax1.fill_between(z, mean_9keV - error_9keV, mean_9keV + error_9keV, facecolor='chartreuse', edgecolor='chartreuse', zorder=3, alpha=0.8,label=r'm$_{\rm wdm} = 9$ keV')
ax1.plot(z,(1.0 - tanH_model(7.68 - 0.79,z)),':',color='maroon',label=r'Planck 1-$\sigma$',zorder=5)
ax1.plot(z,(1.0 - tanH_model(7.68 + 0.79,z)),':',color='maroon',zorder=5)
ax1.errorbar(6.3,0.79,0.04,uplims=True,marker='o', color='gold',markersize=7,label=r'Dark Pixel 2015',zorder=6)
ax1.errorbar(6.5,0.87,0.03,uplims=True,marker='o', color='gold',markersize=7,zorder=6)
ax1.errorbar(6.7,0.94,0.09,uplims=True,marker='o', color='gold',markersize=7,zorder=6)
ax1.errorbar(6.1,0.69,0.06,uplims=True,marker='o', color='gold',markersize=7,zorder=6)
ax1.errorbar(5.9,0.11,0.06,uplims=True,marker='o',markersize=7,label=r'Dark Pixel 2023',zorder=6)
ax1.errorbar(6.9,0.4,0.1,uplims=False,lolims=True,marker='s',markersize=7,label=r'Ly$\alpha$ Fraction',zorder=6)
ax1.errorbar(6.6,0.5,0.1,uplims=True,lolims=False,marker='*',markersize=7,label=r'LAE Clustering',zorder=6)
ax1.errorbar(7.,0.7,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J0252-0503',zorder=6)
ax1.errorbar(7.5,0.39,yerr=np.array([[0.13,0.22]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1007+2115',zorder=6)
ax1.errorbar(7.1,0.4,yerr=np.array([[0.19,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1120+0641',zorder=6)
ax1.errorbar(7.5,0.21,yerr=np.array([[0.19,0.17]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928a',zorder=6)
ax1.errorbar(7.6,0.56,yerr=np.array([[0.18,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928b',zorder=6)
ax1.errorbar(7.4,0.60,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928c',zorder=6)
ax1.errorbar(7.29,0.49,yerr=np.array([[0.11,0.11]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'Combined quasars',zorder=6)
ax1.errorbar(7.0,0.59,yerr=np.array([[0.15,0.11]]).T,xerr=0.5,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWa',zorder=6)
ax1.errorbar(7.6,0.88,yerr=np.array([[0.10,0.05]]).T,xerr=0.6,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWb',zorder=6)
ax1.errorbar(8.0,0.76,yerr=0.22,xerr=0.6,uplims=False,lolims=True,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWc',zorder=6)
ax1.set_xlim(5.5,10)
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'Neutral hydrogen fraction',fontsize=14)
ax1.legend(loc='center left',bbox_to_anchor=(1.,0.5))
plt.savefig('mh_global_history.pdf',bbox_inches="tight")
#plt.savefig('global_history.png',bbox_inches="tight")
plt.show()


# ************* The xe fraction to compare with Planck's plot *************************

  
  
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(z,tanH_model(7.68,z),color='black',label=r'Fit.')
ax1.plot(z,tanH_model(7.68 - 0.79,z),color='magenta',label=r'lower')
ax1.plot(z,tanH_model(7.68 + 0.79,z),color='cyan',label=r'upper')
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'$x_e(z)$',fontsize=14)
ax1.set_xlim(5.5,20)
ax1.legend(loc='best')
plt.savefig('planck2018.pdf',bbox_inches="tight")
plt.show()
# transition to cross power plot
# grab data
#z, k, cross_3keV_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_3keV.txt', unpack=True)
#z, k, cross_4keV_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_4keV.txt', unpack=True)
#z, k, cross_6keV_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_6keV.txt', unpack=True)
#z, k, cross_9keV_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_9keV.txt', unpack=True)
## and fiducial
#z, k, cross_cdm_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_cdm.txt', unpack=True)
## interpolate for easiness

# now need to grab Catalina's data
z, k, cross_r1_cdm = np.loadtxt('../data/21cmFAST/cross_21cm_r1_cdm_s8.txt', unpack=True)
z, k, cross_r2_cdm = np.loadtxt('../data/21cmFAST/cross_21cm_r2_cdm_s8.txt', unpack=True)
z, k, cross_r3_cdm = np.loadtxt('../data/21cmFAST/cross_21cm_r3_cdm_s8.txt', unpack=True)
z, k, cross_r4_cdm = np.loadtxt('../data/21cmFAST/cross_21cm_r4_cdm_s8.txt', unpack=True)
# 3keV
z, k, cross_r1_3keV = np.loadtxt('../data/21cmFAST/cross_21cm_r1_3keV_s8.txt', unpack=True)
z, k, cross_r2_3keV = np.loadtxt('../data/21cmFAST/cross_21cm_r2_3keV_s8.txt', unpack=True)
z, k, cross_r3_3keV = np.loadtxt('../data/21cmFAST/cross_21cm_r3_3keV_s8.txt', unpack=True)
z, k, cross_r4_3keV = np.loadtxt('../data/21cmFAST/cross_21cm_r4_3keV_s8.txt', unpack=True)
# 4keV
z, k, cross_r1_4keV = np.loadtxt('../data/21cmFAST/cross_21cm_r1_4keV_s8.txt', unpack=True)
z, k, cross_r2_4keV = np.loadtxt('../data/21cmFAST/cross_21cm_r2_4keV_s8.txt', unpack=True)
z, k, cross_r3_4keV = np.loadtxt('../data/21cmFAST/cross_21cm_r3_4keV_s8.txt', unpack=True)
z, k, cross_r4_4keV = np.loadtxt('../data/21cmFAST/cross_21cm_r4_4keV_s8.txt', unpack=True)
# 6keV
z, k, cross_r1_6keV = np.loadtxt('../data/21cmFAST/cross_21cm_r1_6keV_s8.txt', unpack=True)
z, k, cross_r2_6keV = np.loadtxt('../data/21cmFAST/cross_21cm_r2_6keV_s8.txt', unpack=True)
z, k, cross_r3_6keV = np.loadtxt('../data/21cmFAST/cross_21cm_r3_6keV_s8.txt', unpack=True)
z, k, cross_r4_6keV = np.loadtxt('../data/21cmFAST/cross_21cm_r4_6keV_s8.txt', unpack=True)
# 9keV
z, k, cross_r1_9keV = np.loadtxt('../data/21cmFAST/cross_21cm_r1_9keV_s8.txt', unpack=True)
z, k, cross_r2_9keV = np.loadtxt('../data/21cmFAST/cross_21cm_r2_9keV_s8.txt', unpack=True)
z, k, cross_r3_9keV = np.loadtxt('../data/21cmFAST/cross_21cm_r3_9keV_s8.txt', unpack=True)
z, k, cross_r4_9keV = np.loadtxt('../data/21cmFAST/cross_21cm_r4_9keV_s8.txt', unpack=True)
z = np.unique(z)
k = np.unique(k)
# get errors and average
cross_mean_cdm = ave(cross_r1_cdm, cross_r2_cdm, cross_r3_cdm, cross_r4_cdm)
cross_mean_3keV = ave(cross_r1_3keV, cross_r2_3keV, cross_r3_3keV, cross_r4_3keV)
cross_mean_4keV = ave(cross_r1_4keV, cross_r2_4keV, cross_r3_4keV, cross_r4_4keV)
cross_mean_6keV = ave(cross_r1_6keV, cross_r2_6keV, cross_r3_6keV, cross_r4_6keV)
cross_mean_9keV = ave(cross_r1_9keV, cross_r2_9keV, cross_r3_9keV, cross_r4_9keV)
cross_error_cdm = std(cross_r1_cdm, cross_r2_cdm, cross_r3_cdm, cross_r4_cdm)
cross_error_3keV = std(cross_r1_3keV, cross_r2_3keV, cross_r3_3keV, cross_r4_3keV)
cross_error_4keV = std(cross_r1_4keV, cross_r2_4keV, cross_r3_4keV, cross_r4_4keV)
cross_error_6keV = std(cross_r1_6keV, cross_r2_6keV, cross_r3_6keV, cross_r4_6keV)
cross_error_9keV = std(cross_r1_9keV, cross_r2_9keV, cross_r3_9keV, cross_r4_9keV)
# interpolate
p_mean_cdm = interp2d(z, k, cross_mean_cdm, kind='cubic')
p_mean_3keV = interp2d(z, k, cross_mean_3keV, kind='cubic')
p_mean_4keV = interp2d(z, k, cross_mean_4keV, kind='cubic')
p_mean_6keV = interp2d(z, k, cross_mean_6keV, kind='cubic')
p_mean_9keV = interp2d(z, k, cross_mean_9keV, kind='cubic')
p_error_cdm = interp2d(z, k, cross_error_cdm, kind='cubic')
p_error_3keV = interp2d(z, k, cross_error_3keV, kind='cubic')
p_error_4keV = interp2d(z, k, cross_error_4keV, kind='cubic')
p_error_6keV = interp2d(z, k, cross_error_6keV, kind='cubic')
p_error_9keV = interp2d(z, k, cross_error_9keV, kind='cubic')

k_test = 0.12

plot_cross = np.zeros((len(z),5))
plot_e_cross = np.zeros((len(z),5))
for i in range(0,len(z)):
    plot_cross[i,4] = p_mean_3keV(z[i], k_test)
    plot_cross[i,3] = p_mean_4keV(z[i], k_test)
    plot_cross[i,2] = p_mean_6keV(z[i], k_test)
    plot_cross[i,1] = p_mean_9keV(z[i], k_test)
    plot_cross[i,0] = p_mean_cdm(z[i], k_test)
    plot_e_cross[i,0] = p_error_cdm(z[i], k_test)
    plot_e_cross[i,4] = p_error_3keV(z[i], k_test)
    plot_e_cross[i,3] = p_error_4keV(z[i], k_test)
    plot_e_cross[i,2] = p_error_6keV(z[i], k_test)
    plot_e_cross[i,1] = p_error_9keV(z[i], k_test)
    
# and plot
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)

for i in range(n):
    ax1.fill_between(z, plot_cross[:,i] - plot_e_cross[:,i], plot_cross[:,i] + plot_e_cross[:,i], facecolor = colors[i], edgecolor=colors[i], label=labels[i])
ax1.grid(linestyle='dotted')
ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
ax1.legend(loc=4)
plt.savefig('cross_power.pdf',bbox_inches="tight")
#plt.savefig('cross_power.png',bbox_inches="tight")
plt.show()

# and for minihalos
# now need to grab Catalina's data
z, k, mh_cross_r1_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r1_cdm.txt', unpack=True)
z, k, mh_cross_r2_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r2_cdm.txt', unpack=True)
z, k, mh_cross_r3_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r3_cdm.txt', unpack=True)
z, k, mh_cross_r4_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r4_cdm.txt', unpack=True)
# 3keV
z, k, mh_cross_r1_3keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r1_wdm3.txt', unpack=True)
z, k, mh_cross_r2_3keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r2_wdm3.txt', unpack=True)
z, k, mh_cross_r3_3keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r3_wdm3.txt', unpack=True)
z, k, mh_cross_r4_3keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r4_wdm3.txt', unpack=True)
# 4keV
z, k, mh_cross_r1_4keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r1_wdm4.txt', unpack=True)
z, k, mh_cross_r2_4keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r2_wdm4.txt', unpack=True)
z, k, mh_cross_r3_4keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r3_wdm4.txt', unpack=True)
z, k, mh_cross_r4_4keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r4_wdm4.txt', unpack=True)
# 6keV
z, k, mh_cross_r1_6keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r1_wdm6.txt', unpack=True)
z, k, mh_cross_r2_6keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r2_wdm6.txt', unpack=True)
z, k, mh_cross_r3_6keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r3_wdm6.txt', unpack=True)
z, k, mh_cross_r4_6keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r4_wdm6.txt', unpack=True)
# 9keV
z, k, mh_cross_r1_9keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r1_wdm9.txt', unpack=True)
z, k, mh_cross_r2_9keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r2_wdm9.txt', unpack=True)
z, k, mh_cross_r3_9keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r3_wdm9.txt', unpack=True)
z, k, mh_cross_r4_9keV = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_mh_r4_wdm9.txt', unpack=True)
z = np.unique(z)
k = np.unique(k)

# need the reference model
znmh, knmh, nmh_cross_r1_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_nmh_r1.txt', unpack=True)
znmh, knmh, nmh_cross_r2_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_nmh_r2.txt', unpack=True)
znmh, knmh, nmh_cross_r3_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_nmh_r3.txt', unpack=True)
znmh, knmh, nmh_cross_r4_cdm = np.loadtxt('../data/21cmFAST/mini_halos/cross_21cmfast_nmh_r4.txt', unpack=True)

nmh_cross_mean_cdm = ave(nmh_cross_r1_cdm, nmh_cross_r2_cdm, nmh_cross_r3_cdm, nmh_cross_r4_cdm)
nmh_cross_error_cdm = std(nmh_cross_r1_cdm, nmh_cross_r2_cdm, nmh_cross_r3_cdm, nmh_cross_r4_cdm)
# get errors and average
mh_cross_mean_cdm = ave(mh_cross_r1_cdm, mh_cross_r2_cdm, mh_cross_r3_cdm, mh_cross_r4_cdm)
mh_cross_mean_3keV = ave(mh_cross_r1_3keV, mh_cross_r2_3keV, mh_cross_r3_3keV, mh_cross_r4_3keV)
mh_cross_mean_4keV = ave(mh_cross_r1_4keV, mh_cross_r2_4keV, mh_cross_r3_4keV, mh_cross_r4_4keV)
mh_cross_mean_6keV = ave(mh_cross_r1_6keV, mh_cross_r2_6keV, mh_cross_r3_6keV, mh_cross_r4_6keV)
mh_cross_mean_9keV = ave(mh_cross_r1_9keV, mh_cross_r2_9keV, mh_cross_r3_9keV, mh_cross_r4_9keV)
mh_cross_error_cdm = std(mh_cross_r1_cdm, mh_cross_r2_cdm, mh_cross_r3_cdm, mh_cross_r4_cdm)
mh_cross_error_3keV = std(mh_cross_r1_3keV, mh_cross_r2_3keV, mh_cross_r3_3keV, mh_cross_r4_3keV)
mh_cross_error_4keV = std(mh_cross_r1_4keV, mh_cross_r2_4keV, mh_cross_r3_4keV, mh_cross_r4_4keV)
mh_cross_error_6keV = std(mh_cross_r1_6keV, mh_cross_r2_6keV, mh_cross_r3_6keV, mh_cross_r4_6keV)
mh_cross_error_9keV = std(mh_cross_r1_9keV, mh_cross_r2_9keV, mh_cross_r3_9keV, mh_cross_r4_9keV)
# interpolate
nmh_p_mean_cdm = interp2d(z, k, nmh_cross_mean_cdm, kind='cubic')
mh_p_mean_cdm = interp2d(z, k, mh_cross_mean_cdm, kind='cubic')
mh_p_mean_3keV = interp2d(z, k, mh_cross_mean_3keV, kind='cubic')
mh_p_mean_4keV = interp2d(z, k, mh_cross_mean_4keV, kind='cubic')
mh_p_mean_6keV = interp2d(z, k, mh_cross_mean_6keV, kind='cubic')
mh_p_mean_9keV = interp2d(z, k, mh_cross_mean_9keV, kind='cubic')
nmh_p_error_cdm = interp2d(z, k, nmh_cross_error_cdm, kind='cubic')
mh_p_error_cdm = interp2d(z, k, mh_cross_error_cdm, kind='cubic')
mh_p_error_3keV = interp2d(z, k, mh_cross_error_3keV, kind='cubic')
mh_p_error_4keV = interp2d(z, k, mh_cross_error_4keV, kind='cubic')
mh_p_error_6keV = interp2d(z, k, mh_cross_error_6keV, kind='cubic')
mh_p_error_9keV = interp2d(z, k, mh_cross_error_9keV, kind='cubic')

k_test = 0.12

mh_plot_cross = np.zeros((len(z),6))
mh_plot_e_cross = np.zeros((len(z),6))
for i in range(0,len(z)):
    mh_plot_cross[i,5] = mh_p_mean_3keV(z[i], k_test)
    mh_plot_cross[i,4] = mh_p_mean_4keV(z[i], k_test)
    mh_plot_cross[i,3] = mh_p_mean_6keV(z[i], k_test)
    mh_plot_cross[i,2] = mh_p_mean_9keV(z[i], k_test)
    mh_plot_cross[i,1] = mh_p_mean_cdm(z[i], k_test)
    mh_plot_cross[i,0] = nmh_p_mean_cdm(z[i], k_test)
    mh_plot_e_cross[i,0] = nmh_p_error_cdm(z[i], k_test)
    mh_plot_e_cross[i,1] = mh_p_error_cdm(z[i], k_test)
    mh_plot_e_cross[i,5] = mh_p_error_3keV(z[i], k_test)
    mh_plot_e_cross[i,4] = mh_p_error_4keV(z[i], k_test)
    mh_plot_e_cross[i,3] = mh_p_error_6keV(z[i], k_test)
    mh_plot_e_cross[i,2] = mh_p_error_9keV(z[i], k_test)

# and plot
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)

for i in range(mh_n):
    ax1.fill_between(z, mh_plot_cross[:,i] - mh_plot_e_cross[:,i], mh_plot_cross[:,i] + mh_plot_e_cross[:,i], facecolor = mh_colors[i], edgecolor=mh_colors[i], label=mh_labels[i])
ax1.grid(linestyle='dotted')
ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
ax1.legend(loc=4)
plt.savefig('mh_cross_power.pdf',bbox_inches="tight")
#plt.savefig('cross_power.png',bbox_inches="tight")
plt.show()


#sys.exit()


#plot_3keV = np.zeros(len(z))
#plot_4keV = np.zeros(len(z))
#plot_6keV = np.zeros(len(z))
#plot_9keV = np.zeros(len(z))
#plot_cdm = np.zeros(len(z))
#plot_e_cdm = np.zeros(len(z))
#plot_e_3keV = np.zeros(len(z))
#plot_e_4keV = np.zeros(len(z))
#plot_e_6keV = np.zeros(len(z))
#plot_e_9keV = np.zeros(len(z))
#for i in range(0,len(z)):
#    plot_3keV[i] = p_mean_3keV(z[i], k_test)
#    plot_4keV[i] = p_mean_4keV(z[i], k_test)
#    plot_6keV[i] = p_mean_6keV(z[i], k_test)
#    plot_9keV[i] = p_mean_9keV(z[i], k_test)
#    plot_cdm[i] = p_mean_cdm(z[i], k_test)
#    plot_e_cdm[i] = p_error_cdm(z[i], k_test)
#    plot_e_3keV[i] = p_error_3keV(z[i], k_test)
#    plot_e_4keV[i] = p_error_4keV(z[i], k_test)
#    plot_e_6keV[i] = p_error_6keV(z[i], k_test)
#    plot_e_9keV[i] = p_error_9keV(z[i], k_test)
#
## and plot
#fig = plt.figure(figsize=(6,6))
#ax1 = fig.add_subplot(111)
#ax1.fill_between(z, plot_cdm - plot_e_cdm, plot_cdm + plot_e_cdm, facecolor='black', edgecolor='black', alpha=0.8, label=r'CDM')
#ax1.fill_between(z, plot_3keV - plot_e_3keV, plot_3keV + plot_e_3keV, facecolor='purple', edgecolor='purple', alpha=0.8,label=r'm$_{\rm wdm} = 3$ keV')
#ax1.fill_between(z, plot_4keV - plot_e_4keV, plot_4keV + plot_e_4keV, facecolor='salmon', edgecolor='salmon', alpha=0.8,label=r'm$_{\rm wdm} = 4$ keV')
#ax1.fill_between(z, plot_6keV - plot_e_6keV, plot_6keV + plot_e_6keV, facecolor='aqua', edgecolor='aqua', alpha=0.8,label=r'm$_{\rm wdm} = 6$ keV')
#ax1.fill_between(z, plot_9keV - plot_e_9keV, plot_9keV + plot_e_9keV, facecolor='chartreuse', edgecolor='chartreuse', alpha=0.8,label=r'm$_{\rm wdm} = 9$ keV')
##ax1.plot(z,plot_cdm,color='black',label='CDM')
##ax1.plot(z,plot_3keV,color='purple',label=r'm$_{\rm wdm} = 3$ keV')
##ax1.plot(z,plot_4keV,color='salmon',label=r'm$_{\rm wdm} = 4$ keV')
##ax1.plot(z,plot_6keV,color='aqua',label=r'm$_{\rm wdm} = 6$ keV')
##ax1.plot(z,plot_9keV,color='chartreuse',label=r'm$_{\rm wdm} = 9$ keV')
#ax1.grid(linestyle='dotted')
#ax1.set_xlim(5.5,12)
#ax1.set_xlabel(r'Redshift',fontsize=14)
#ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
#ax1.legend(loc='best')
#plt.savefig('cross_power.pdf',bbox_inches="tight")
##plt.savefig('cross_power.png',bbox_inches="tight")
#plt.show()









#sys.exit()
# let's do midpoint, 75 percent, and 25 percent of reionization

# Also, I want a plot as function of wavenumber
z_50_cdm = history_cdm(0.5)
z_50_3keV = history_3keV(0.5)
z_50_4keV = history_4keV(0.5)
z_50_6keV = history_6keV(0.5)
z_50_9keV = history_9keV(0.5)
kplot_mid = np.zeros((len(k), 5))
kerror_mid = np.zeros((len(k), 5))
for i in range(0,len(k)):
    kplot_mid[i,0] = p_mean_cdm(z_50_cdm, k[i])
    kplot_mid[i,1] = p_mean_9keV(z_50_9keV, k[i])
    kplot_mid[i,2] = p_mean_6keV(z_50_6keV, k[i])
    kplot_mid[i,3] = p_mean_4keV(z_50_4keV, k[i])
    kplot_mid[i,4] = p_mean_3keV(z_50_3keV, k[i])
    kerror_mid[i,0] = p_error_cdm(z_50_cdm, k[i])
    kerror_mid[i,1] = p_error_9keV(z_50_9keV, k[i])
    kerror_mid[i,2] = p_error_6keV(z_50_6keV, k[i])
    kerror_mid[i,3] = p_error_4keV(z_50_4keV, k[i])
    kerror_mid[i,4] = p_error_3keV(z_50_3keV, k[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
for i in range(n):
    ax1.fill_between(k, kplot_mid[:,i] - kerror_mid[:,i], kplot_mid[:,i] + kerror_mid[:,i], facecolor = colors[i], edgecolor=colors[i], label=labels[i])
ax1.set_xscale('log')
ax1.grid(linestyle='dotted')
ax1.legend(loc='best')
ax1.set_title(r'$x_{\rm HI} = 0.50$',fontsize=14)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
plt.savefig('cross_power_func_k_50.pdf',bbox_inches="tight")
plt.show()


nmh_z_50_cdm = nmh_history_cdm(0.5)
mh_z_50_cdm  = mh_history_cdm(0.5)
mh_z_50_3keV = mh_history_3keV(0.5)
mh_z_50_4keV = mh_history_4keV(0.5)
mh_z_50_6keV = mh_history_6keV(0.5)
mh_z_50_9keV = mh_history_9keV(0.5)
mh_kplot_mid = np.zeros((len(k), 6))
mh_kerror_mid = np.zeros((len(k), 6))
for i in range(0,len(k)):
    mh_kplot_mid[i,0] = nmh_p_mean_cdm(mh_z_50_cdm, k[i])
    mh_kplot_mid[i,1] = mh_p_mean_cdm(mh_z_50_cdm, k[i])
    mh_kplot_mid[i,2] = mh_p_mean_9keV(mh_z_50_9keV, k[i])
    mh_kplot_mid[i,3] = mh_p_mean_6keV(mh_z_50_6keV, k[i])
    mh_kplot_mid[i,4] = mh_p_mean_4keV(mh_z_50_4keV, k[i])
    mh_kplot_mid[i,5] = mh_p_mean_3keV(mh_z_50_3keV, k[i])
    mh_kerror_mid[i,0] = nmh_p_error_cdm(mh_z_50_cdm, k[i])
    mh_kerror_mid[i,1] = mh_p_error_cdm(mh_z_50_cdm, k[i])
    mh_kerror_mid[i,2] = mh_p_error_9keV(mh_z_50_9keV, k[i])
    mh_kerror_mid[i,3] = mh_p_error_6keV(mh_z_50_6keV, k[i])
    mh_kerror_mid[i,4] = mh_p_error_4keV(mh_z_50_4keV, k[i])
    mh_kerror_mid[i,5] = mh_p_error_3keV(mh_z_50_3keV, k[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
for i in range(mh_n):
    ax1.fill_between(k, mh_kplot_mid[:,i] - mh_kerror_mid[:,i], mh_kplot_mid[:,i] + mh_kerror_mid[:,i], facecolor = mh_colors[i], edgecolor=mh_colors[i], label=mh_labels[i])
ax1.set_xscale('log')
ax1.grid(linestyle='dotted')
ax1.legend(loc='best')
ax1.set_title(r'$x_{\rm HI} = 0.50$',fontsize=14)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
plt.savefig('mh_cross_power_func_k_50.pdf',bbox_inches="tight")
plt.show()


""" now computing the 25 per cent """

z_50_cdm = history_cdm(0.25)
z_50_3keV = history_3keV(0.25)
z_50_4keV = history_4keV(0.25)
z_50_6keV = history_6keV(0.25)
z_50_9keV = history_9keV(0.25)
kplot_mid = np.zeros((len(k), 5))
kerror_mid = np.zeros((len(k), 5))
for i in range(0,len(k)):
    kplot_mid[i,0] = p_mean_cdm(z_50_cdm, k[i])
    kplot_mid[i,1] = p_mean_9keV(z_50_9keV, k[i])
    kplot_mid[i,2] = p_mean_6keV(z_50_6keV, k[i])
    kplot_mid[i,3] = p_mean_4keV(z_50_4keV, k[i])
    kplot_mid[i,4] = p_mean_3keV(z_50_3keV, k[i])
    kerror_mid[i,0] = p_error_cdm(z_50_cdm, k[i])
    kerror_mid[i,1] = p_error_9keV(z_50_9keV, k[i])
    kerror_mid[i,2] = p_error_6keV(z_50_6keV, k[i])
    kerror_mid[i,3] = p_error_4keV(z_50_4keV, k[i])
    kerror_mid[i,4] = p_error_3keV(z_50_3keV, k[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
for i in range(n):
    ax1.fill_between(k, kplot_mid[:,i] - kerror_mid[:,i], kplot_mid[:,i] + kerror_mid[:,i], facecolor = colors[i], edgecolor=colors[i], label=labels[i])
ax1.set_xscale('log')
ax1.grid(linestyle='dotted')
ax1.legend(loc='best')
ax1.set_title(r'$x_{\rm HI} = 0.25$',fontsize=14)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
plt.savefig('cross_power_func_k_25.pdf',bbox_inches="tight")
plt.show()



nmh_z_50_cdm  = nmh_history_cdm(0.25)
mh_z_50_cdm  = mh_history_cdm(0.25)
mh_z_50_3keV = mh_history_3keV(0.25)
mh_z_50_4keV = mh_history_4keV(0.25)
mh_z_50_6keV = mh_history_6keV(0.25)
mh_z_50_9keV = mh_history_9keV(0.25)
mh_kplot_mid = np.zeros((len(k), 6))
mh_kerror_mid = np.zeros((len(k), 6))
for i in range(0,len(k)):
    mh_kplot_mid[i,0] = nmh_p_mean_cdm(nmh_z_50_cdm, k[i])
    mh_kplot_mid[i,1] = mh_p_mean_cdm(mh_z_50_cdm, k[i])
    mh_kplot_mid[i,2] = mh_p_mean_9keV(mh_z_50_9keV, k[i])
    mh_kplot_mid[i,3] = mh_p_mean_6keV(mh_z_50_6keV, k[i])
    mh_kplot_mid[i,4] = mh_p_mean_4keV(mh_z_50_4keV, k[i])
    mh_kplot_mid[i,5] = mh_p_mean_3keV(mh_z_50_3keV, k[i])
    mh_kerror_mid[i,0] = nmh_p_error_cdm(nmh_z_50_cdm, k[i])
    mh_kerror_mid[i,1] = mh_p_error_cdm(mh_z_50_cdm, k[i])
    mh_kerror_mid[i,2] = mh_p_error_9keV(mh_z_50_9keV, k[i])
    mh_kerror_mid[i,3] = mh_p_error_6keV(mh_z_50_6keV, k[i])
    mh_kerror_mid[i,4] = mh_p_error_4keV(mh_z_50_4keV, k[i])
    mh_kerror_mid[i,5] = mh_p_error_3keV(mh_z_50_3keV, k[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
for i in range(mh_n):
    ax1.fill_between(k, mh_kplot_mid[:,i] - mh_kerror_mid[:,i], mh_kplot_mid[:,i] + mh_kerror_mid[:,i], facecolor = mh_colors[i], edgecolor=mh_colors[i], label=mh_labels[i])
ax1.set_xscale('log')
ax1.grid(linestyle='dotted')
ax1.legend(loc='best')
ax1.set_title(r'$x_{\rm HI} = 0.25$',fontsize=14)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
plt.savefig('mh_cross_power_func_k_25.pdf',bbox_inches="tight")
plt.show()

""" now computing the 75 per cent """

z_50_cdm = history_cdm(0.75)
z_50_3keV = history_3keV(0.75)
z_50_4keV = history_4keV(0.75)
z_50_6keV = history_6keV(0.75)
z_50_9keV = history_9keV(0.75)
kplot_mid = np.zeros((len(k), 5))
kerror_mid = np.zeros((len(k), 5))
for i in range(0,len(k)):
    kplot_mid[i,0] = p_mean_cdm(z_50_cdm, k[i])
    kplot_mid[i,1] = p_mean_9keV(z_50_9keV, k[i])
    kplot_mid[i,2] = p_mean_6keV(z_50_6keV, k[i])
    kplot_mid[i,3] = p_mean_4keV(z_50_4keV, k[i])
    kplot_mid[i,4] = p_mean_3keV(z_50_3keV, k[i])
    kerror_mid[i,0] = p_error_cdm(z_50_cdm, k[i])
    kerror_mid[i,1] = p_error_9keV(z_50_9keV, k[i])
    kerror_mid[i,2] = p_error_6keV(z_50_6keV, k[i])
    kerror_mid[i,3] = p_error_4keV(z_50_4keV, k[i])
    kerror_mid[i,4] = p_error_3keV(z_50_3keV, k[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
for i in range(n):
    ax1.fill_between(k, kplot_mid[:,i] - kerror_mid[:,i], kplot_mid[:,i] + kerror_mid[:,i], facecolor = colors[i], edgecolor=colors[i], label=labels[i])
ax1.set_xscale('log')
ax1.grid(linestyle='dotted')
ax1.legend(loc='best')
ax1.set_title(r'$x_{\rm HI} = 0.75$',fontsize=14)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
plt.savefig('cross_power_func_k_75.pdf',bbox_inches="tight")
plt.show()



nmh_z_50_cdm  = nmh_history_cdm(0.75)
mh_z_50_cdm  = mh_history_cdm(0.75)
mh_z_50_3keV = mh_history_3keV(0.75)
mh_z_50_4keV = mh_history_4keV(0.75)
mh_z_50_6keV = mh_history_6keV(0.75)
mh_z_50_9keV = mh_history_9keV(0.75)
mh_kplot_mid = np.zeros((len(k), 6))
mh_kerror_mid = np.zeros((len(k), 6))
for i in range(0,len(k)):
    mh_kplot_mid[i,0] = nmh_p_mean_cdm(nmh_z_50_cdm, k[i])
    mh_kplot_mid[i,1] = mh_p_mean_cdm(mh_z_50_cdm, k[i])
    mh_kplot_mid[i,2] = mh_p_mean_9keV(mh_z_50_9keV, k[i])
    mh_kplot_mid[i,3] = mh_p_mean_6keV(mh_z_50_6keV, k[i])
    mh_kplot_mid[i,4] = mh_p_mean_4keV(mh_z_50_4keV, k[i])
    mh_kplot_mid[i,5] = mh_p_mean_3keV(mh_z_50_3keV, k[i])
    mh_kerror_mid[i,0] = nmh_p_error_cdm(nmh_z_50_cdm, k[i])
    mh_kerror_mid[i,1] = mh_p_error_cdm(mh_z_50_cdm, k[i])
    mh_kerror_mid[i,2] = mh_p_error_9keV(mh_z_50_9keV, k[i])
    mh_kerror_mid[i,3] = mh_p_error_6keV(mh_z_50_6keV, k[i])
    mh_kerror_mid[i,4] = mh_p_error_4keV(mh_z_50_4keV, k[i])
    mh_kerror_mid[i,5] = mh_p_error_3keV(mh_z_50_3keV, k[i])

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
for i in range(mh_n):
    ax1.fill_between(k, mh_kplot_mid[:,i] - mh_kerror_mid[:,i], mh_kplot_mid[:,i] + mh_kerror_mid[:,i], facecolor = mh_colors[i], edgecolor=mh_colors[i], label=mh_labels[i])
ax1.set_xscale('log')
ax1.grid(linestyle='dotted')
ax1.legend(loc='best')
ax1.set_title(r'$x_{\rm HI} = 0.75$',fontsize=14)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
plt.savefig('mh_cross_power_func_k_75.pdf',bbox_inches="tight")
plt.show()







sys.exit()








# and plot
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(k,plot_cdm,color='black')
ax1.fill_between(k, plot_cdm - plot_e_cdm, plot_cdm + plot_e_cdm, facecolor='black', edgecolor='black', alpha=0.8, label=r'CDM')
ax1.plot(k,plot_3keV,color='purple')
ax1.fill_between(k, plot_3keV - plot_e_3keV, plot_3keV + plot_e_3keV, facecolor='purple', edgecolor='purple', alpha=0.8,label=r'm$_{\rm wdm} = 3$ keV')
ax1.plot(k,plot_4keV,color='salmon')
ax1.fill_between(k, plot_4keV - plot_e_4keV, plot_4keV + plot_e_4keV, facecolor='salmon', edgecolor='salmon', alpha=0.8,label=r'm$_{\rm wdm} = 4$ keV')
ax1.plot(k,plot_6keV,color='aqua')
ax1.fill_between(k, plot_6keV - plot_e_6keV, plot_6keV + plot_e_6keV, facecolor='aqua', edgecolor='aqua', alpha=0.8,label=r'm$_{\rm wdm} = 6$ keV')
ax1.plot(k,plot_9keV,color='chartreuse')
ax1.fill_between(k, plot_9keV - plot_e_9keV, plot_9keV + plot_e_9keV, facecolor='chartreuse', edgecolor='chartreuse', alpha=0.8,label=r'm$_{\rm wdm} = 9$ keV')
ax1.grid(linestyle='dotted')
ax1.set_title(r'$x_{\rm HI} = 0.50$',fontsize=14)
#ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
ax1.legend(loc='best')
plt.savefig('cross_power_func_k_50.pdf',bbox_inches="tight")
#plt.savefig('cross_power.png',bbox_inches="tight")
plt.show()


z_25_cdm = history_cdm(0.25)
z_25_3keV = history_3keV(0.25)
z_25_4keV = history_4keV(0.25)
z_25_6keV = history_6keV(0.25)
z_25_9keV = history_9keV(0.25)
plot_3keV = np.zeros(len(k))
plot_4keV = np.zeros(len(k))
plot_6keV = np.zeros(len(k))
plot_9keV = np.zeros(len(k))
plot_cdm = np.zeros(len(k))
for i in range(0,len(k)):
    plot_3keV[i] = p_mean_3keV(z_25_3keV, k[i])
    plot_4keV[i] = p_mean_4keV(z_25_4keV, k[i])
    plot_6keV[i] = p_mean_6keV(z_25_6keV, k[i])
    plot_9keV[i] = p_mean_9keV(z_25_9keV, k[i])
    plot_cdm[i] = p_mean_cdm(z_25_cdm, k[i])
    plot_e_cdm[i] = p_error_cdm(z_25_cdm, k[i])
    plot_e_3keV[i] = p_error_3keV(z_25_3keV, k[i])
    plot_e_4keV[i] = p_error_3keV(z_25_4keV, k[i])
    plot_e_6keV[i] = p_error_3keV(z_25_6keV, k[i])
    plot_e_9keV[i] = p_error_3keV(z_25_9keV, k[i])
# and plot
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(k,plot_cdm,color='black')
ax1.fill_between(k, plot_cdm - plot_e_cdm, plot_cdm + plot_e_cdm, facecolor='black', edgecolor='black', alpha=0.8, label=r'CDM')
ax1.plot(k,plot_3keV,color='purple')
ax1.fill_between(k, plot_3keV - plot_e_3keV, plot_3keV + plot_e_3keV, facecolor='purple', edgecolor='purple', alpha=0.8,label=r'm$_{\rm wdm} = 3$ keV')
ax1.plot(k,plot_4keV,color='salmon')
ax1.fill_between(k, plot_4keV - plot_e_4keV, plot_4keV + plot_e_4keV, facecolor='salmon', edgecolor='salmon', alpha=0.8,label=r'm$_{\rm wdm} = 4$ keV')
ax1.plot(k,plot_6keV,color='aqua')
ax1.fill_between(k, plot_6keV - plot_e_6keV, plot_6keV + plot_e_6keV, facecolor='aqua', edgecolor='aqua', alpha=0.8,label=r'm$_{\rm wdm} = 6$ keV')
ax1.plot(k,plot_9keV,color='chartreuse')
ax1.fill_between(k, plot_9keV - plot_e_9keV, plot_9keV + plot_e_9keV, facecolor='chartreuse', edgecolor='chartreuse', alpha=0.8,label=r'm$_{\rm wdm} = 9$ keV')
ax1.grid(linestyle='dotted')
ax1.set_title(r'$x_{\rm HI} = 0.25$',fontsize=14)
#ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
ax1.legend(loc='best')
plt.savefig('cross_power_func_k_25.pdf',bbox_inches="tight")
#plt.savefig('cross_power.png',bbox_inches="tight")
plt.show()

z_75_cdm = history_cdm(0.75)
z_75_3keV = history_3keV(0.75)
z_75_4keV = history_4keV(0.75)
z_75_6keV = history_6keV(0.75)
z_75_9keV = history_9keV(0.75)
plot_3keV = np.zeros(len(k))
plot_4keV = np.zeros(len(k))
plot_6keV = np.zeros(len(k))
plot_9keV = np.zeros(len(k))
plot_cdm = np.zeros(len(k))
for i in range(0,len(k)):
    plot_3keV[i] = p_mean_3keV(z_75_3keV, k[i])
    plot_4keV[i] = p_mean_4keV(z_75_4keV, k[i])
    plot_6keV[i] = p_mean_6keV(z_75_6keV, k[i])
    plot_9keV[i] = p_mean_9keV(z_75_9keV, k[i])
    plot_cdm[i] = p_mean_cdm(z_75_cdm, k[i])
    plot_e_cdm[i] = p_error_cdm(z_75_cdm, k[i])
    plot_e_3keV[i] = p_error_3keV(z_75_3keV, k[i])
    plot_e_4keV[i] = p_error_3keV(z_75_4keV, k[i])
    plot_e_6keV[i] = p_error_3keV(z_75_6keV, k[i])
    plot_e_9keV[i] = p_error_3keV(z_75_9keV, k[i])
# and plot
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(k,plot_cdm,color='black')
ax1.fill_between(k, plot_cdm - plot_e_cdm, plot_cdm + plot_e_cdm, facecolor='black', edgecolor='black', alpha=0.8, label=r'CDM')
ax1.plot(k,plot_3keV,color='purple')
ax1.fill_between(k, plot_3keV - plot_e_3keV, plot_3keV + plot_e_3keV, facecolor='purple', edgecolor='purple', alpha=0.8,label=r'm$_{\rm wdm} = 3$ keV')
ax1.plot(k,plot_4keV,color='salmon')
ax1.fill_between(k, plot_4keV - plot_e_4keV, plot_4keV + plot_e_4keV, facecolor='salmon', edgecolor='salmon', alpha=0.8,label=r'm$_{\rm wdm} = 4$ keV')
ax1.plot(k,plot_6keV,color='aqua')
ax1.fill_between(k, plot_6keV - plot_e_6keV, plot_6keV + plot_e_6keV, facecolor='aqua', edgecolor='aqua', alpha=0.8,label=r'm$_{\rm wdm} = 6$ keV')
ax1.plot(k,plot_9keV,color='chartreuse')
ax1.fill_between(k, plot_9keV - plot_e_9keV, plot_9keV + plot_e_9keV, facecolor='chartreuse', edgecolor='chartreuse', alpha=0.8,label=r'm$_{\rm wdm} = 9$ keV')
ax1.grid(linestyle='dotted')
ax1.set_title(r'$x_{\rm HI} = 0.75$',fontsize=14)
#ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Wavenumber',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
ax1.legend(loc='best')
plt.savefig('cross_power_func_k_75.pdf',bbox_inches="tight")
#plt.savefig('cross_power.png',bbox_inches="tight")
plt.show()
