#%% 
# This file used to generate trainset and testset (.h5 files) for training ResNet
import os
import pandas as pd
import numpy as np
import h5py
import dataio as dio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


#%%
def load_data(filepath):
    df = dio.readfile(filepath)  # Assuming dio.readfile works like pd.read_fits
    trainset = df.sample(frac=0.8, random_state=24) # default seed: 2024
    testset = df.drop(trainset.index)
    return trainset, testset


def process_single_data(cid, richness, backg, z, som_path):
    path = Path(som_path % cid)
    if path.exists():
        img = pd.read_csv(path, sep='\s+', header=None).to_numpy(dtype=np.float32)
        if img.max() == 0.:
            richness = 0.0
            backg = 0
    else:
        img = np.zeros([200, 200], dtype=np.float32)
        richness = 0.0
        backg = 0

    return img, richness, backg, z, cid


def process_data(dataset, output_path, som_path):
    num_samples = len(dataset)
    X = np.zeros([num_samples, 1, 200, 200], dtype=np.float32)
    Y = np.zeros(num_samples, dtype=np.float32)
    Z = np.zeros(num_samples, dtype=np.float32)
    B = np.zeros(num_samples, dtype=np.float32)
    ID = np.zeros(num_samples, dtype=np.int32)

    dataset = dataset.set_index('concat_id')

    tasks = []
    with ProcessPoolExecutor() as executor:
        for cid, row in dataset.iterrows():
            richness = row['richness']
            z = row['z']
            backg = row['backg']
            tasks.append(executor.submit(process_single_data, cid, richness, backg, z, som_path))

        for idx, future in enumerate(tasks):
            img, richness, backg, z, cid = future.result()
            X[idx, 0, :, :] = img
            Y[idx] = richness
            B[idx] = backg
            Z[idx] = z
            ID[idx] = cid

            if idx % 10000 == 0:
                print(f"{idx} / {num_samples} finished")


    with h5py.File(output_path, 'w') as f:
        f.create_dataset('x', data=X)
        f.create_dataset('y', data=Y)
        f.create_dataset('backg', data=B)
        f.create_dataset('z', data=Z)
        f.create_dataset('id', data=ID)





# %% change .h5 --> .fits
path = '../data/resnet_train_data_concat_withid.h5'
df = dio.readfile(path)

path = '../traindata/resnet_train_data_concat_withid.fits'
dio.savefile(df, path)


#%%
path = '../traindata/resnet_train_data_concat_withid.fits'
trainset, testset = load_data(path)

output_path = '../traindata/trainset_seed24.fits'
dio.savefile(trainset, output_path)
output_path = '../traindata/testset_seed24.fits'
dio.savefile(testset, output_path)

train_output_path = '../traindata/trainset_seed24_sigma0.05_1mpc.h5'
test_output_path = '../traindata/testset_seed24_sigma0.05_1mpc.h5'
som_path = '../data/train_data/gsm_0.05/%06d.dat'

process_data(trainset, train_output_path, som_path)
process_data(testset, test_output_path, som_path)


#%%
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
font = {'size': 18}
font1 = {'size': 25}
plt.rc('font', **font)
# 设置xtick和ytick的方向：in、out、inout
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1

nbins = 50
bins = np.linspace(0, 1, nbins)
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
res = np.histogram(trainset[trainset.z<1].z,bins=nbins)
plt.step(res[1][:-1], res[0], color='k',linestyle='--', linewidth=2)
res = np.histogram(testset[testset.z<1].z,bins=nbins)
plt.step(res[1][:-1], res[0], color='k',linestyle='-', linewidth=2)
plt.xlabel('Redshift', fontdict=font1)
plt.ylabel('Number', fontdict=font1)
plt.xlim(-0.05,0.85)
plt.ylim(0,6e3)
plt.xticks([0,0.2,0.4,0.6,0.8],[0,0.2,0.4,0.6,0.8])
plt.yticks([0,2e3,4e3,6e3],[0,2e3,4e3,6e3])
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(['Training set', 'Test set'],frameon=False,fontsize=17)

ax2 = fig.add_subplot(132)
res = np.histogram(trainset.dered_r,bins=nbins)
plt.step(res[1][:-1], res[0], color='k',linestyle='--', linewidth=2)
res = np.histogram(testset.dered_r,bins=nbins)
plt.step(res[1][:-1], res[0], color='k',linestyle='-', linewidth=2)
plt.xlabel(r'$r$-band magnitude', fontdict=font1)
plt.ylabel('Number', fontdict=font1)
plt.xlim(13,23)
plt.ylim(0,1.1e4)
plt.xticks([14,16,18,20,22],[14,16,18,20,22])
plt.yticks([5e3, 1e4], [5e3, 1e4])
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax3 = fig.add_subplot(133)
res = np.histogram(trainset.richness,bins=nbins)
plt.step(res[1][:-1], res[0], color='k',linestyle='--', linewidth=2)
res = np.histogram(testset.richness,bins=nbins)
plt.step(res[1][:-1], res[0], color='k',linestyle='-', linewidth=2)
plt.xlabel('WHL richness', fontdict=font1)
plt.ylabel('Number', fontdict=font1)
plt.xscale('log')
plt.yscale('log')
plt.xlim([5,200])
plt.ylim(1e1,1e5)
plt.xticks([10,20,50,100,200],[10,20,50,100,200])
plt.yticks([1e2,1e3,1e4,1e5], [1e2,1e3,1e4,1e5])
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
