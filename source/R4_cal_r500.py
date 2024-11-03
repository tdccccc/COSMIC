#%%
import numpy as np
import pandas as pd
from scipy import stats

import tools as t
import dataio as dio

#%%
def fit_to_whl_richness(pred_richness, richness):
    res = stats.linregress(pred_richness, richness)
    slope = res.slope
    intercept = res.intercept
    stderr = res.stderr
    intercept_stderr = res.intercept_stderr
    return res, slope, intercept, stderr, intercept_stderr


def richness2whlrichness(richness):
    k = 1.02
    b = 0.06
    # delta_k = 0.01
    # delta_b = 0.01
    whlrichness = richness**(k) * 10**(-b)
    return whlrichness

def richness2m500(whlrichness):
    k = 1.08
    b = 1.37
    # delta_k = 0.02
    # delta_b = 0.02
    m500 = whlrichness**k * 10**(-b)
    return m500



#%% 1st estimate
path = '../output/BCG_cand.fits'
cand = dio.readfile(path)
cand['whl_richness'] = richness2whlrichness(cand['pred_richness'])
cand['pred_m500'] = richness2m500(cand['whl_richness'])
cand['pred_r500'] = t.m5002r500(cand['pred_m500'], cand['z'])

path = '../output/BCG_cand.fits'
dio.savefile(cand, path)

