import warnings
warnings.filterwarnings('ignore')
import os
import h5py
import random
import pandas as pd
import numpy as np


def get_color(df):
    df['g-i'] = df.dered_g - df.dered_i
    df['g-r'] = df.dered_g - df.dered_r
    df['u-r'] = df.dered_u - df.dered_r
    df['r-i'] = df.dered_r - df.dered_i
    df['i-z'] = df.dered_i - df.dered_z
    df['psf-model'] = df.psfMag_i - df.modelMag_i
    df['modelMagErr_ur'] = np.sqrt(df.modelMagErr_u**2 + df.modelMagErr_r**2)
    df['modelMagErr_gi'] = np.sqrt(df.modelMagErr_g**2 + df.modelMagErr_i**2)
    df['modelMagErr_ri'] = np.sqrt(df.modelMagErr_r**2 + df.modelMagErr_i**2)
    df['modelMagErr_iz'] = np.sqrt(df.modelMagErr_i**2 + df.modelMagErr_z**2)
    return df


def sdss_data_clean(df):
    ind = (df.dered_u>0) & (df.dered_g>0) & (df.dered_r>0)
    ind = ind & (df.dered_i>0) & (df.dered_z>0)
    ind = ind & ((df.modelMagErr_u>0)) & (df.modelMagErr_g>0)
    ind = ind & (df.modelMagErr_r>0) & (df.modelMagErr_i>0)
    ind = ind & (df.modelMagErr_z>0) 
    ind = ind & (df.deVAB_r>0) & (df.deVABErr_r>0)
    ind = ind & (df.deVRad_r>0) & (df.deVRadErr_r>0)
    ind = ind & (df.psfMag_i>0)
    df = df[ind]
    return df


def phot_err_limit(df):
    ind = (df.zErr > 0) & (df.z > 0) & (df.zErr < 0.04) 
    ind = ind & (df.modelMagErr_u <2.8)
    ind = ind & (df.modelMagErr_g < 0.5) & (df.modelMagErr_r < 0.1)
    ind = ind & (df.modelMagErr_i < 0.06) & (df.modelMagErr_z < 0.14)
    ind = ind & (df.deVRadErr_r < 5) & ((df.deVABErr_r < 0.8))
    df = df[ind]
    return df


def remove_stars(df):
    ind =  (df['psf-model'] > (-0.125*df.dered_i + 2.9))
    df = df[ind]
    return df


def xgb_input_normalization(df):
    sdss_path = '../data/sdss_clean.h5'
    sdss = pd.read_hdf(sdss_path,key='data',mode='r')
    # normalization
    for col in df.columns:
        stddev = sdss[col].std()
        mean = sdss[col].mean()
        df[col] = (df[col] - mean) / stddev
    return df


def xgboost_data_preprocessing():
    sdss_path = '../../data/DownloadCatlog/sdss_dr17_all_galaxy/SDSS_DR17_Galaxy_SpecCM_Specconcat_rmQsoStar_WHL15CM.h5'
    sdss = pd.read_hdf(sdss_path,key='data',mode='r')
    ind = (sdss.ra>150.)&(sdss.ra<160.)&(sdss.dec>0.)&(sdss.dec<20.)
    gcap = sdss[ind].copy()
    # 数据清理，保证数据可用
    sdss = sdss_data_clean(sdss)
    gcap = sdss_data_clean(gcap)
    # 限制测光误差
    sdss = phot_err_limit(sdss)
    gcap = phot_err_limit(gcap)
    # 计算星系的颜色
    sdss = get_color(sdss)
    gcap = get_color(gcap)
    # 排除可能的恒星，即psf-model ~ 0源
    sdss = remove_stars(sdss)
    gcap = remove_stars(gcap)
    gcap['zWarning'] = -1
    gcap['absMagR'] = 0
    gcap['num_mem'] = 0
    path1 = '../data/sdss_clean.h5'
    path2 = '../data/gcap_clean.h5'
    sdss.to_hdf(path1,key='data',mode='w',format='table')
    gcap.to_hdf(path2,key='data',mode='w',format='table')
    return sdss, gcap


# preprocess learning data (select parms & normalization) for XGBoost
def make_xgboost_train_data(richness_threshold):
    path1 = '../data/train_data/xgboost_train_x.h5'
    path2 = '../data/train_data/xgboost_train_y.h5'
    path3 = '../data/train_data/xgboost_test_x.h5'
    path4 = '../data/train_data/xgboost_test_y.h5'

    random.seed(2022)
    sdss_path = '../data/sdss_clean.h5'
    sdss = pd.read_hdf(sdss_path,key='data',mode='r')
    # choose needed columns
    label = (sdss.richness > richness_threshold).astype(int)
    sdss['label'] = label
    label2 = ((sdss.label==0)&(sdss.richness>0)).astype(int)
    sdss['label2'] = label2
    ind = (sdss.label2!=1) & (sdss.phot_z>0)
    sdss = sdss[ind]
    sdss['z'] = sdss.phot_z.values

    choose_cols = ['z', 'dered_r','u-r','g-i','r-i','i-z', 'modelMagErr_r',
                    'modelMagErr_ur', 'modelMagErr_gi', 'modelMagErr_ri', 'modelMagErr_iz',
                    'deVRadErr_r','deVRad_r','deVAB_r','deVABErr_r']
    input_sdss = sdss[choose_cols]
    # normalization
    # input_sdss = xgb_input_normalization(input_sdss)
    # get 1:1 pos/neg samples
    input_sdss['label'] = label
    pos_samples = input_sdss[input_sdss['label'] == 1]
    neg_samples = input_sdss[input_sdss['label'] == 0]
    num_samples = pos_samples.shape[0]
    total_samples = pd.concat([pos_samples.sample(n=num_samples),
                               neg_samples.sample(n=num_samples)])
    # split train & test set
    train_size = int(total_samples.shape[0] * 0.8)
    train_ind = random.sample(list(total_samples.index),train_size)
    test_ind = list(set(total_samples.index).difference(set(train_ind)))
    # generate x & y
    total_y = total_samples['label']
    total_x = total_samples.drop('label', axis=1)
    # train & test data
    train_x = total_x.loc[train_ind,:]
    train_y = total_y.loc[train_ind]
    test_x  = total_x.loc[test_ind,:]
    test_y  = total_y.loc[test_ind]
    # save data
    train_x.to_hdf(path1,key='data',mode='w',format='table')
    train_y.to_hdf(path2,key='data',mode='w',format='table')
    test_x.to_hdf(path3,key='data',mode='w',format='table')
    test_y.to_hdf(path4,key='data',mode='w',format='table')
    # 输出未归一化数据
    train_set = total_samples.loc[train_ind,:]
    test_set = total_samples.loc[test_ind,:]
    train_set.to_hdf('../data/train_data/xgboost_trainset.h5',
                     key='data',mode='w',format='table')
    test_set.to_hdf('../data/train_data/xgboost_testset.h5',
                     key='data',mode='w',format='table')
    return train_x,train_y,test_x,test_y


if __name__=='__main__':
    sdss, gcap = xgboost_data_preprocessing()
    train_x,train_y,test_x,test_y = make_xgboost_train_data(richness_threshold=10.0)