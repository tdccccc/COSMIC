# BCG classification
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import xgboost as xgb
import numpy as np

import dataio as dio
import XGBoost_train_data_process as xdp


def clean(df):
    df = xdp.sdss_data_clean(df)
    df = xdp.phot_err_limit(df)
    df = xdp.get_color(df)
    df = xdp.remove_stars(df)
    return df

def load_XGBoost_input(df):
    choose_cols = ['z', 'dered_r','u-r','g-i','r-i','i-z', 'modelMagErr_r',
                    'modelMagErr_ur', 'modelMagErr_gi', 'modelMagErr_ri', 'modelMagErr_iz',
                    'deVRadErr_r','deVRad_r','deVAB_r','deVABErr_r']
    input_df = df[choose_cols]
    return input_df, df


# using a new sky area (default north galatic cap) to test XGBoost model
def BCG_classificaton(df, threshold):
    input_df, df = load_XGBoost_input(df)
    model = xgb.Booster()
    model.load_model(fname='../model/XGBoost_model.model') 
    predprob = model.predict(xgb.DMatrix(input_df))
    # print('There are', len(input_df), 'galaxies.')
    # print('Threshold is set as %.2f.'%(threshold))
    print('Number of BCG candidates: %d'%(len(predprob[predprob>threshold])))
    predictions = predprob > threshold
    BCG_cand = df[predictions]
    BCG_cand['ind_ref'] = range(len(BCG_cand))
    return BCG_cand.reset_index()


if __name__=='__main__':
    path = '../data/test_data.fits'
    df = dio.readfile(path)

    idx = df.spec_z>0
    df.loc[idx, 'z'] = df.loc[idx, 'spec_z']

    # select R.A.: 154.5-155.5 deg., Dec: 4.5-5.5 deg.
    idx = (df.ra > 154.5) & (df.ra < 155.5)
    idx &= (df.dec > 4.5) & (df.dec < 5.5)
    df = df[idx]

    df = clean(df)
    BCG_cand = BCG_classificaton(df, threshold=0.4)

    output_path = '../output/BCG_cand.fits'
    dio.savefile(BCG_cand, output_path)

