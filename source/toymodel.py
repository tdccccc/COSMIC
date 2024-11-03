#%%
import dataio as dio
import numpy as np
import pandas as pd

#%%
path = '../../apjs_reply/data/gcap_all.h5'
df = dio.readfile(path)


#%%
# select R.A. 154-156deg., Dec. 4-6deg. for toymodel
idx = (df.ra > 154.) & (df.ra <  156.)
idx &= (df.dec > 4.) & (df.dec < 6.)
tab = df[idx]
path = '../data/test_data.fits'
dio.savefile(tab, path)


#%%
path = 'f_gsm/plot_data/galaxy43s.dat'
arr = np.loadtxt(path)

idx = (arr[:,1] > 154.) & (arr[:,1] < 156.)
idx &= (arr[:,2] > 4.) & (arr[:,2] < 6.)
tab = arr[idx]

path = '../data/backg_gal.dat'
np.savetxt(path, tab, 
           fmt='%7d %12.5f %10.5f %8.2f %8.4f %8.4f %8.2f %3d %3d',
           header='', 
           comments='', 
           delimiter=' ')

#%%
path = '../data/merge_gcap.dat'
tab = np.loadtxt(path)

cols = ['id','ra','dec','mag_r','z','zErr','absMagR','flag','is_spec']
df = pd.DataFrame(tab, columns=cols)

path = '../data/backg_gal.fits'
dio.savefile(df, path)