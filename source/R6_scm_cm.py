# This file used to execute self-cross-match and cross-match

import numpy as np
import pandas as pd
import load_catalogs as lc
import dataio as dio
import tools as t

# self cross match step 2 BCG candidates, to find close candidates
def self_cross_match(df):
    # determine cluster redshift
    df['z_cl'] = df['z_mem_mean'].values
    ind = df['z_specmem_mean']>0
    df.loc[ind, 'z_cl'] = df.loc[ind, 'z_specmem_mean']

    
    df['ddz'] = df['z'].apply(t.disofz)
    # cross label
    df['r500_ang'] = df['pred_r500']/df.ddz.values
    # drop indexies
    df['scm_rich_drop'] = -1
    df['scm_magr_drop'] = -1
    df['scm_Nr500_drop'] = -1
    df['have_nb'] = -1
    ang_delta = 0.5
    ang = 3.14159/180.0
    for gidx in df.index:
        gdata = df.loc[gidx,:]
        gra = gdata.ra
        gdec = gdata.dec
        gz = gdata.z_cl
        ind = (df.ra < (gra+ang_delta)) & (df.ra > (gra-ang_delta))
        ind &= ((df.dec < (gdec+ang_delta)) & (df.dec > (gdec-ang_delta)))
        list_near = df[ind]

        dra = list_near.ra - gra
        ddec = list_near.dec - gdec
        dz = np.abs(list_near.z_cl - gz)
        dis = np.sqrt(np.abs(dra)**2*np.cos(gdec*ang)**2 + np.abs(ddec)**2)

        dis_condition = dis < 1.5 * gdata['r500_ang']
        z_condition = dz < 0.06 * (1 + gz)

        # 先找到满足距离和红移条件的星系
        valid_near = list_near[dis_condition & z_condition]

        if not valid_near.empty:
            df.loc[gidx, 'have_nb'] = 1

            # 再根据满足的条件查找更大的富度、亮度和Nr500
            rich_condition = (gdata.pred_richness < valid_near.pred_richness).any()
            mag_r_condition = (-1. * gdata.dered_r < -1. * valid_near.dered_r).any()
            Nr500_condition = (gdata.N_r500 < valid_near.N_r500).any()

            if mag_r_condition:
                df.loc[gidx, 'scm_magr_drop'] = 1

            if rich_condition:
                df.loc[gidx, 'scm_rich_drop'] = 1

            if Nr500_condition:
                df.loc[gidx, 'scm_Nr500_drop'] = 1

    df = df.sort_values(by='ind_ref', ascending=True)
    return df


def generate_cluster(df):
    ind = (df['scm_rich_drop'] == -1)
    ind &= df.pred_richness>=10
    ind &= df.N_r500 >= 6 # 成员星系>=5
    return df[ind]


# cross-match between BCG candidate and galaxy cluster catalogs
def cat_cross(cand, gcc, gcc_name, cand_name='All', 
              print_tab=True, save_MatchedCluster=False):
    ang = np.pi/180.0
    # add label
    cand['id_'+gcc_name] = -1
    cand['if_matched_2arcsec_'+gcc_name] = -1
    cand['if_matched_1halfr500_'+gcc_name] = -1
    # num
    Ncross_2arcsec = 0
    Ncross_1halfr500 = 0
    # collect missed and matched clusters
    MatchedCluster = gcc.copy()
    MatchedCluster['id_%s'%(cand_name)] = -1
    MatchedCluster['2arcsec_matched'] = -1
    MatchedCluster['1halfr500_matched'] = -1
    for cid in gcc.index:
        # 获取gcc源的基本坐标
        cdata = gcc.loc[cid,:]
        cra = cdata['ra']
        cdec = cdata['dec']
        cz = cdata['z']
        # difference between each cordinate
        dra = cand.ra - cra
        ddec = cand.dec - cdec
        dz = np.abs(cand['z_cl'] - cz)
        # cand源与目标源的坐标差
        z_threshold = 0.05*(1 + cz)
        # distance between the galaxy cluster and predicted galaxies
        dis = np.sqrt(np.abs(dra)**2*np.cos(cdec*ang)**2 + np.abs(ddec)**2)
        # cross match in 1.5r500
        mind1half = (dis < 1.5*cand['r500_ang']) & (dz < z_threshold)
        if np.sum(mind1half) > 0:
            Ncross_1halfr500 += 1
            MatchedCluster.loc[cid, '1halfr500_matched'] = 1
            cand.loc[mind1half, 'if_matched_1halfr500_'+gcc_name] = 1

            # cross match in 2 arcsec
            mind2arcsec = (dis < 2/60/60) & (dz < z_threshold)
            if np.sum(mind2arcsec) > 0:
                Ncross_2arcsec += 1
                MatchedCluster.loc[cid, '2arcsec_matched'] = 1
                cand.loc[mind2arcsec, 'if_matched_2arcsec_'+gcc_name] = 1
    
            # matched nearest source
            nearest_source_index = cand.index[dz<z_threshold][np.argmin(dis[dz<z_threshold])]
            MatchedCluster.loc[cid, 'id_%s'%(cand_name)] = nearest_source_index
            cand.loc[nearest_source_index, 'id_'+gcc_name] = cid
    
    if save_MatchedCluster:
        MatchedCluster.to_csv('../output/MatchedCluster_%s.csv' %
                        (gcc_name), float_format='%.6f')
        
    if print_tab:
        Ngcc = len(gcc)
        print('%s & %d & %d & %.2f\\%% & %d & %.2f\\%% \\\\' %
               (gcc_name, Ngcc, 
                Ncross_2arcsec, Ncross_2arcsec/Ngcc*100, 
                Ncross_1halfr500, Ncross_1halfr500/Ngcc*100))
    return cand,MatchedCluster

   
# choose data from maxBCG 
def choose_maxbcg():
    # gcc = lc.load_maxbcg()
    path = '../data/clus_cat/gcap_cm_catalogs/MaxBCG_gcap_input_matched.csv'
    gcc = pd.read_csv(path)
    gcc.rename(columns={'RAdeg':'ra','DEdeg':'dec'},inplace=True)
    gcc.loc[:,'z'] = gcc.loc[:,'zph']
    ind = gcc.zsp>0
    gcc.loc[ind,'z'] = gcc.loc[ind,'zsp']
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)
    gcc = gcc[gcc_choose]
    return gcc

# choose data from AMF
def choose_AMF():
    gcc = lc.load_AMF()
    gcc.rename(columns={'Radeg':'ra','Dedeg':'dec','zc':'z'},inplace=True)
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)
    gcc = gcc[gcc_choose]
    return gcc

# choose data from GMBCG
def choose_gmbcg():
    # gcc = lc.load_gmbcg()
    path = '../data/clus_cat/gcap_cm_catalogs/GMBCG_gcap_input_matched.csv'
    gcc = pd.read_csv(path)
    gcc.rename(columns={'RAdeg':'ra','DEdeg':'dec'},inplace=True)
    gcc.loc[:,'z'] = gcc.loc[:,'zph']
    ind = gcc.zsp>0
    gcc.loc[ind,'z'] = gcc.loc[ind,'zsp']
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)
    gcc = gcc[gcc_choose]
    return gcc
    
# choose data from redMaPPer
def choose_redmapper():
    # gcc = lc.load_redmapper_cat_dr8()
    path = '../data/clus_cat/gcap_cm_catalogs/redMaPPer_gcap_input_matched.csv'
    gcc = pd.read_csv(path)
    gcc.rename(columns={'RAdeg':'ra','DEdeg':'dec','zlambda':'zph','zspec':'zsp'},
              inplace=True)
    gcc.loc[:,'z'] = gcc.loc[:,'zph']
    ind = gcc.zsp>0
    gcc.loc[ind,'z'] = gcc.loc[ind,'zsp']
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)
    gcc = gcc[gcc_choose]
    return gcc

# choose data from WHL15
def choose_whl():
    # gcc = lc.load_WHL()
    path = '../data/clus_cat/gcap_cm_catalogs/WHL15_gcap_input_matched.csv'
    gcc = pd.read_csv(path)
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)
    gcc = gcc[gcc_choose]
    return gcc

def choose_Yang2021():
    gcc = lc.load_Yang2021()
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)&(gcc.z<0.65)
    gcc = gcc[gcc_choose]
    return gcc

def choose_zou2021():
    # gcc = lc.load_zou2021()
    path = '../data/clus_cat/gcap_cm_catalogs/Zou2021_gcap_input_matched.csv'
    gcc = pd.read_csv(path)
    gcc.rename(columns={'RA_BCG':'ra','DEC_BCG':'dec','PZ_BCG':'phot_z',
                       'SZ_BCG':'spec_z'}, inplace=True)
    gcc.loc[:,'z'] = gcc.loc[:,'phot_z']
    ind = gcc.spec_z>0
    gcc.loc[ind,'z'] = gcc.loc[ind,'spec_z']
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)&(gcc.z<0.65)
    gcc = gcc[gcc_choose]
    return gcc

def choose_CluMPR():
    # gcc = lc.load_CluMPR()
    path = '../data/clus_cat/gcap_cm_catalogs/CluMPR_gcap_input_matched.csv'
    gcc = pd.read_csv(path)
    gcc.rename(columns={'RA_central':'ra','DEC_central':'dec'},
            inplace=True)
    gcc.loc[:,'z'] = gcc.loc[:,'z_median_central']
    ind = gcc.spec_z>0
    gcc.loc[ind,'z'] = gcc.loc[ind,'spec_z']
    gcc_choose = (gcc.ra>150)&(gcc.ra<160)&(gcc.dec>0)&(gcc.dec<20)&(gcc.z<0.65)
    gcc = gcc[gcc_choose]
    return gcc

# choose catalog by name   
def choose_catalog(name):
    if name == 'MaxBCG':
        gcc = choose_maxbcg()
    elif name == 'GMBCG':
        gcc = choose_gmbcg()
    elif name == 'AMF':
        gcc =  choose_AMF()
    elif name == 'redMaPPer':
        gcc = choose_redmapper()
    elif name == 'WHL15':
        gcc = choose_whl()
    elif name == 'Yang2021':
        gcc = choose_Yang2021()
    elif name == 'Zou2021':
        gcc = choose_zou2021()
    elif name == 'CluMPR':
        gcc = choose_CluMPR()
    else:
        gcc = []
        raise Exception("Invalid catalog name!")
    return gcc

# cross match with all catalogs
def compare_all(df):
    cat_list = ['MaxBCG','GMBCG','AMF','redMaPPer','WHL15',
                'Yang2021','Zou2021','CluMPR']
    for gcc_name in cat_list:
        gcc = choose_catalog(gcc_name)
        df,_ = cat_cross(df, gcc, gcc_name, save_MatchedCluster=False)
    df['in_cat'] = ((df.if_matched_1halfr500_WHL15 > 0)
                            | (df.if_matched_1halfr500_AMF > 0)
                            | (df.if_matched_1halfr500_GMBCG > 0)
                            | (df.if_matched_1halfr500_MaxBCG > 0)
                            | (df.if_matched_1halfr500_redMaPPer > 0)).astype('int')
    
    ind = df.in_cat == 0
    new = df[ind]
    print('Number of all clusters: %d' % (len(df)))
    print('Number of new clusters: %d' % (len(new)))
    return df, new



if __name__ == '__main__':
    # scm
    path = '../output/BCG_cand.fits'
    df = dio.readfile(path)
    BCG_cand_scm = self_cross_match(df)
    dio.savefile(BCG_cand_scm, path)

    # set threshold and generate fullcatalog
    path = '../output/BCG_cand.fits'
    BCG_cand_scm = dio.readfile(path)
    cluster = generate_cluster(BCG_cand_scm)
    cluster_path = '../output/cluster.fits'
    dio.savefile(cluster, cluster_path)

    # # cm
    # input_path = '../output/cluster.fits'
    # cluster = dio.readfile(input_path)
    # cluster, new = compare_all(cluster)
    # new_path = '../output/new_cluster.fits'
    # dio.savefile(new, new_path)
    # cluster_path = '../output/cluster.fits'
    # dio.savefile(cluster, cluster_path)
