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
z, xH_3keV = pylab.loadtxt('./data/21cmFAST/xH_test_3keV.txt',unpack=True)
# grab the fiducial
z, xH_cdm = pylab.loadtxt('./data/21cmFAST/xH_test_cdm.txt',unpack=True)
# and then we plot them
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(z,xH_cdm,color='black',label=r'cdm',zorder=4)
ax1.plot(z,xH_3keV,color='purple',label=r'm$_{\rm wdm} = 3$ keV',zorder=4)
ax1.axvline(x=5.90,linestyle='dashed',color='navy')
ax1.plot(z,(1.0 - tanH_model(7.68 - 0.79,z)),':',color='maroon',label=r'Planck 1-$\sigma$',zorder=5)
ax1.plot(z,(1.0 - tanH_model(7.68 + 0.79,z)),':',color='maroon',zorder=5)
ax1.errorbar(5.9,0.11,0.06,uplims=True,marker='o',markersize=7,label=r'Dark Pixel',zorder=6)
ax1.errorbar(6.9,0.4,0.1,uplims=False,lolims=True,marker='s',markersize=7,label=r'Ly$\alpha$ Fraction',zorder=6)
ax1.errorbar(6.6,0.5,0.1,uplims=True,lolims=False,marker='*',markersize=7,label=r'LAE Clustering',zorder=6)
ax1.errorbar(7.1,0.4,yerr=np.array([[0.19,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'ULAS J1120+0641',zorder=6)
ax1.errorbar(7.5,0.21,yerr=np.array([[0.19,0.17]]).T,uplims=False,marker='v',markersize=7,capthick=2,capsize=4,label=r'ULAS J1342+0928a',zorder=6)
ax1.errorbar(7.6,0.56,yerr=np.array([[0.18,0.21]]).T,uplims=False,marker='^',markersize=7,capthick=2,capsize=4,label=r'ULAS J1342+0928b',zorder=6)
ax1.errorbar(7.4,0.60,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='<',markersize=7,capthick=2,capsize=4,label=r'ULAS J1342+0928c',zorder=6)
ax1.errorbar(7.0,0.59,yerr=np.array([[0.15,0.11]]).T,xerr=0.5,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWa',zorder=6)
ax1.errorbar(7.6,0.88,yerr=np.array([[0.10,0.05]]).T,xerr=0.6,uplims=False,marker='D',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWb',zorder=6)
ax1.errorbar(8.0,0.76,yerr=0.22,xerr=0.6,uplims=False,lolims=True,marker='X',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWc',zorder=6)
ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'Neutral hydrogen fraction',fontsize=14)
ax1.legend(loc='best')
plt.savefig('global_history.pdf',bbox_inches="tight")
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
z, k, cross_3keV_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_3keV.txt', unpack=True)
# and fiducial
z, k, cross_cdm_a = np.loadtxt('./data/21cmFAST/cross_21cm_test_cdm.txt', unpack=True)
# interpolate for easiness
z = np.unique(z)
k = np.unique(k)
cross_3kev = interp2d(z, k, cross_3keV_a, kind='cubic')
cross_cdm = interp2d(z, k, cross_cdm_a, kind='cubic')
plot_3keV = np.zeros(len(z))
plot_cdm = np.zeros(len(z))
for i in range(0,len(z)):
    plot_3keV[i] = cross_3kev(z[i], 0.12)
    plot_cdm[i] = cross_cdm(z[i], 0.12)

    
# and plot
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.plot(z,plot_cdm,color='black',label=r'cdm')
ax1.plot(z,plot_3keV,color='purple',label=r'm$_{\rm wdm} = 3$ keV')
#ax1.set_xlim(5.5,12)
ax1.set_xlabel(r'Redshift',fontsize=14)
ax1.set_ylabel(r'$\frac{k^3}{2\pi^2} P_{m,x_{HI}}$',fontsize=14)
ax1.legend(loc='best')
plt.savefig('cross_power.pdf',bbox_inches="tight")
#plt.savefig('cross_power.png',bbox_inches="tight")
plt.show()
