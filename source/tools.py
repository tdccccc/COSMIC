import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const

def Eofz(z):
    Omega_Lambda = 0.7
    Omega_m = 0.3
    Omega_k = 0
    temp1 = Omega_m*(1+z)**3
    temp2 = Omega_k*(1+z)**2
    return np.sqrt(temp1 + temp2 + Omega_Lambda)

def disofz(z, rad2ang=True):
    N_step = 100
    dz = z / N_step
    sum = 0
    for i in range(N_step):
        sum += dz / Eofz(dz*i)
    dis = 4285.7*sum/(1+z)
    if rad2ang:
        return dis/57.3
    return dis

def Hofz(z):
    Omega_m0 = 0.3
    Omega_Lambda = 0.7
    H_0 = 70
    Eofz = np.sqrt(Omega_m0*(1+z)**3 + Omega_Lambda)
    return H_0*Eofz

def m5002r500(m500, z):
    pc = const.pc.value
    G = const.G.value
    M_sun = const.M_sun.value
    temp1 = 2*m500*10**(14)*M_sun*G
    temp2 = 500*Hofz(z)**2
    unit1 = 10**6 * pc**2
    unit2 = 10**6 * pc
    temp3 = (temp1 / temp2 * unit1) **(1/3)
    r500 = temp3 / unit2 
    return r500

def specz_slice(sigma_v, z):
    temp1 = sigma_v * (1 + z) * 10**(-5)
    temp2 = 25/3 * 10**(-3) * (1 + z)
    return np.max([temp1, temp2])

def photz_slice(z):
    return 0.04*(1+z) if z <= 0.45 else (0.248*z - 0.0536)

def cal_bias(label, prediction, mode='log', operation='average'):
    if mode=='log':
        dev = np.abs(np.log10(label) - np.log10(prediction))
    if mode=='normal':
        dev = np.abs(label - prediction)
    if operation == 'average':
        res = np.average(dev)
    elif operation == 'median':
        res = np.median(dev)
    print('bias: %.3f'%res)
    return res

def cal_dispersion(label, prediction, mode='log'):
    if mode=='log':
        dev = np.abs(np.log10(label) - np.log10(prediction))
    if mode=='normal':
        dev = np.abs(label - prediction)
    temp1 = np.std(dev)
    temp2 = 1.4*np.median(np.abs(dev))
    print('dispersion(均方差): %.3f'%temp1)
    print('dispersion(1.4倍中值): %.3f'%temp2)
    return temp1,temp2 

