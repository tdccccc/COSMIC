# Caluculate background
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import dataio as dio


# make a table for fortran files to load
def make_BCG_candidate_fortran_data():
    BCG_cand = dio.readfile('../output/BCG_cand.fits')
    BCG_cand['absMagR'] = -1
    choose_cols = ['ind_ref','ra','dec','z','spec_z','dered_r','absMagR','num_mem']
    BCG_cand_fortran_input = BCG_cand[choose_cols]
    BCG_cand_fortran_input['is_spec'] = (BCG_cand.spec_z>0).astype('int')
    BCG_cand_fortran_input['num_mem'] = BCG_cand_fortran_input['num_mem'].astype('int')
    fortran_data = '../output/f_richness/predcen4calc.dat'
    BCG_cand_fortran_input.to_csv(fortran_data,sep=' ',header=False,
                        index=False,float_format="%.6f")


# Wen's program to calculate richness for a given BCG
def calculate_BCG_candidate_WHL15richness():
    path = 'f_richness/'
    exe_files = ['RC1.out','RC2.out']
    fortran_files = ['RC1_cluster_2nd1.f','RC2_back.f']
    # move to fortran source folder
    os.chdir(path)
    for nfile in range(len(exe_files)):
        exef = exe_files[nfile]
        # in linux system, run ./[filename], as follows
        fort = fortran_files[nfile]
        # compile fortran source code
        os.system('gfortran %s -o %s'%(fort,exef))
        # execute
        os.system('./%s'%(exef))
    # back to this folder
    os.chdir('..')


def load_backg(df):
    backg_tab = np.loadtxt('../output/f_richness/RC_back.dat')
    df['backg'] = backg_tab[:, 4]
    return df

if __name__ == '__main__':
    make_BCG_candidate_fortran_data()
    calculate_BCG_candidate_WHL15richness()

    path = '../output/BCG_cand.fits'
    df = dio.readfile(path)
    # df = load_WHL15_richness(df)
    df = load_backg(df)
    output_path = '../output/BCG_cand.fits'
    dio.savefile(df, output_path)
