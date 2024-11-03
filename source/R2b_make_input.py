# Generate SOM and make into ResNet input
import warnings
warnings.filterwarnings('ignore')
import os
import h5py
import time
from pathlib import Path
import numpy as np
import pandas as pd
import dataio as dio
import tools as t
from concurrent.futures import ProcessPoolExecutor



def add_zslice(df):
    df['r_som'] = 1
    df['sig_som'] = 0.05
    df['photz_slice'] = df['z'].apply(t.photz_slice)
    df['specz_slice'] = 25/3 * 10**(-3) * (1 + df['z']) # delta V < 2500 km s^-1 
    return df


def make_fortran_GSM_input(df):
    df = add_zslice(df)

    df_finput = df.loc[:,['ind_ref','ra','dec','z',
                          'r_som', 'sig_som', 
                          'photz_slice', 'specz_slice']]
    output_path = '../output/fgsm_input.dat'
    np.savetxt(fname=output_path, X=df_finput,
               fmt=['%d','%.5f', '%.5f', '%.4f', 
                    '%.1f', '%.3f','%.5f', '%.5f'])
    return df


def calculate_GSM():
    path = 'f_gsm/'
    exe_file = 'make_som'
    fortran_file = 'make_som.f'
    os.chdir(path)
    os.system('gfortran %s -o %s'%(fortran_file,exe_file))
    os.system('./%s'%(exe_file))
    os.chdir('..')



def process_single_data(cid, backg, z, som_path):
    path = Path(som_path % cid)
    if path.exists():
        img = pd.read_csv(path, sep='\s+', header=None).to_numpy(dtype=np.float32)
        if img.max() == 0.:
            backg = 0
    else:
        img = np.zeros([200, 200], dtype=np.float32)
        backg = 0

    return img, backg, z, cid


def process_data(dataset, output_path, som_path):
    num_samples = len(dataset)
    X = np.zeros([num_samples, 1, 200, 200], dtype=np.float32)
    Z = np.zeros(num_samples, dtype=np.float32)
    B = np.zeros(num_samples, dtype=np.float32)
    ID = np.zeros(num_samples, dtype=np.int32)

    dataset = dataset.set_index('ind_ref')

    tasks = []
    with ProcessPoolExecutor() as executor:
        for cid, row in dataset.iterrows():
            z = row['z']
            backg = row['backg']
            tasks.append(executor.submit(process_single_data, cid, backg, z, som_path))

        for idx, future in enumerate(tasks):
            img, backg, z, cid = future.result()
            X[idx, 0, :, :] = img
            B[idx] = backg
            Z[idx] = z
            ID[idx] = cid

            if idx % 10000 == 0:
                print(f"{idx} / {num_samples} finished")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('x', data=X)
        f.create_dataset('backg', data=B)
        f.create_dataset('z', data=Z)
        f.create_dataset('id', data=ID)


if __name__ == '__main__':
    path = '../output/BCG_cand.fits'
    df = dio.readfile(path)

    df = make_fortran_GSM_input(df)
    dio.savefile(df, path)

    calculate_GSM()

    # 继续执行后续代码
    path = '../output/BCG_cand.fits'
    df = dio.readfile(path)
    train_output_path = '../output/ResNet_input.h5'
    som_path = '../output/som/%06d.dat'
    process_data(df, train_output_path, som_path)