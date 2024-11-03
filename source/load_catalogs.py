import os 
import pandas as pd

cat_path = '../data/clus_cat'
down_cat = '../data/down_cat'
plot_cat = 'fortran_gause_smooth/plot_data'

# Load GMBCG Galaxy Cluster Catalog   
def load_gmbcg():
    global cat_path
    tb_path = '%s/gmbcg/gmbcgdr7.dat' %cat_path
    csv_path = '%s/gmbcg/gmbcgdr7.csv' %cat_path
    if os.path.exists(csv_path):
        tb = pd.read_csv(csv_path)
        return tb
    colnames = ['objid','RAdeg','DEdeg','zph','e_zph','zsp',
                'g-r','e_g-r','r-i','e_r-i','umag','gmag','rmag',
                'imag','zmag','e_umag','e_gmag','e_rmag','e_imag',
                'e_zmag','Scl','Ng','Ng_W','q_Ng_W','label','GMBCG']
    tb = pd.read_csv(tb_path,names=colnames,sep='\s+')
    tb.to_csv(csv_path,index=False)
    return tb

# Load RedMaPPer Galaxy Cluster Catalog (Members)   
def load_redmapper_mmb_dr8():
    global cat_path
    tb_path = '%s/redmapper/mmb_dr8.dat' %cat_path
    csv_path = '%s/redmapper/mmb_dr8.csv' %cat_path
    if os.path.exists(csv_path):
        tb = pd.read_csv(csv_path)
        return tb
    colnames = ['ID','RAdeg','DEdeg','R','PMem','Pfree','tLum','tRad','imag',
                'e_imag','umagm','e_umagm','gmagm','e_gmagm','ramgm','e_rmagm',
                'imagm','e_imagm','zmagm','e_zmagm','zspec','objID']
    tb = pd.read_csv(tb_path,names=colnames,sep='|')
    tb.to_csv(csv_path)
    return tb
    
# Load RedMaPPer Galaxy Cluster Catalog (BCGs)  
def load_redmapper_cat_dr8():
    global cat_path
    tb_path = '%s/redmapper/cat_dr8.dat' %cat_path
    csv_path = '%s/redmapper/cat_dr8.csv' %cat_path
    if os.path.exists(csv_path):
        tb = pd.read_csv(csv_path)
        return tb
    colnames = ['ID','Name','RAdeg','DEdeg','zlambda','e_zlambda','lambda',
                'e_lambda','S','zspec','ObjID','imag','e_imag','umagm','e_umagm',
                'gmagm','e_gmagm','rmagm','e_rmagm','imagm','e_imagm','zmagm',
                'e_zmagm','iLum','PCenN','RANdeg','DENdeg','IDN','PZbinN','PZN']
    other =    ['PCen0','PCen1','PCen2','PCen3','PCen4','RA0deg',
                'RA1deg','RA2deg','RA3deg','RA4deg','DE0deg','DE1deg','DE2deg',
                'DE3deg','DE4deg','ID0','ID1','ID2','ID3','ID4','PZbin1','PZbin2',
                'PZbin3','PZbin4','PZbin5','PZbin6','PZbin7','PZbin8','PZbin9',
                'PZbin10','PZbin11','PZbin12','PZbin13','PZbin14','PZbin15',
                'PZbin16','PZbin17','PZbin18','PZbin19','PZbin20','PZbin21',
                'PZ1','PZ2','PZ3','PZ4','PZ5','PZ6','PZ7','PZ8','PZ9','PZ10',
                'PZ11','PZ12','PZ13','PZ14','PZ15','PZ16','PZ17','PZ18','PZ19',
                'PZ20','PZ21']
    print('columns not included:\n',other)
    tb = pd.read_csv(tb_path,names=colnames,sep='|')
    tb.to_csv(csv_path,index=False)
    return tb

# Load MaxBCG Galaxy Cluster Catalog
def load_maxbcg():
    global cat_path
    tb_path = '%s/maxbcg/table1.dat' %cat_path
    csv_path = '%s/maxbcg/table1.csv' %cat_path
    if os.path.exists(csv_path):
        tb = pd.read_csv(csv_path)
        return tb
    colnames = ['RAdeg','DEdeg','zph','zsp','LBr','LBi','LTr','LTi','Ngal','NR200']
    tb = pd.read_csv(tb_path,names=colnames,sep='\s+')
    tb.to_csv(csv_path,index=False)
    return tb

# Load AMF Galaxy Cluster Catalog     
def load_AMF():
    global cat_path
    tb_path = '%s/AMF/gccat.csv' %cat_path
    csv_path = '%s/AMF/gccat_clean.csv' %cat_path
    if os.path.exists(csv_path):
        tb = pd.read_csv(csv_path)
        return tb
    tb = pd.read_csv(tb_path)
    n = 0
    for n in range(len(tb)):
        line = tb.loc[n,:]
        ra_mbcg = line.RA_mbcg.strip()
        if ra_mbcg == '-':
            tb.loc[n,'RA_mbcg'] = ""
        else:
            tb.loc[n,'RA_mbcg'] = ra_mbcg
        de_mbcg = line.DE_mbcg.strip()
        if de_mbcg == '-':
            tb.loc[n,'DE_mbcg'] = ""
        else:
            tb.loc[n,'DE_mbcg'] = de_mbcg
        z_mbcg = line.z_mbcg.strip()
        if z_mbcg == '-':
            tb.loc[n,'z_mbcg'] = ""
        else:
            tb.loc[n,'z_mbcg'] = z_mbcg
        whl = line.WHL.strip()
        tb.loc[n,'WHL'] = whl
        abell = line.Abell.strip()
        tb.loc[n,'Abell'] = abell
    tb.to_csv(csv_path,index = False)
    return tb

# load WHL15 galaxy cluster catalog    
def load_WHL():
    global cat_path
    tb_path = "%s/whl/WHL15_dr14.dat" %cat_path
    csv_path = '%s/whl/WHL15_dr14.csv' %cat_path
    if os.path.exists(csv_path):
        tb = pd.read_csv(csv_path)
        return tb
    colnames = ['id', 'ra', 'dec', 'z', 'dered_r', 'r500', 'richness',
                'num_mem', 'is_spec']
    tb = pd.read_csv(tb_path, names=colnames, index_col='id', sep='\s+')
    tb.to_csv(csv_path)
    return tb

# downloaded dr14 SDSS spec-z data
def load_sdss_dr14():
    global down_cat
    tb = pd.read_csv("%s/sdss_galaxy_all_dr14.csv" %down_cat,
        index_col='objid')
    return tb

# download dr14 north galatic cap data (photo-z)
def load_gcap_dr14():
    global down_cat
    tb = pd.read_csv("%s/gcap_galaxy_all_dr14.csv" %down_cat,
        index_col='objid')
    return tb

def load_Yang2021():
    cols = ['group_id','richness','ra','dec','z','group_mass','group_lum']
    tb = pd.read_csv('../../data/DownloadCatlog/DESI_NGCP_selected.csv',
                      index_col='group_id')
    return tb

def load_desi_sgcp_group():
    cols = ['group_id','richness','ra','dec','z','group_mass','group_lum']
    tb = pd.read_csv('../../data/DownloadCatlog/DESI_SGCP_selected.csv',
                      index_col='group_id')
    return tb

def load_zou2021():
    tb = pd.read_csv('../../data/DownloadCatlog/ZouHU/Zou2021.csv',
                      index_col='CLUSTER_ID')
    return tb

def load_CluMPR():
    tb = pd.read_csv('../../data/DownloadCatlog/CluMPR2023/CLUMPR_DESI.csv',
                     index_col='id')
    return tb