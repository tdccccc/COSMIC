# Calculate Ngal
import numpy as np
import dataio as dio


# 用于获取BCG周围的成员星系相关信息
class MemberGalaxy:
    def __init__(self,input_path,output_path):
        self.ang = 3.14159/180.0
        self.Omega_Lambda = 0.7
        self.Omega_M = 0.3
        self.input_path = input_path
        self.output_path = output_path

    # angular & euclidian distance at different z
    def dis_z(self,z):
        N_step=100
        dz=z/float(N_step)
        temp=0.0
        for j in range(1,N_step+1):
            temp += dz/np.sqrt(self.Omega_Lambda + self.Omega_M*(1+dz*float(j))**3)
        dist=4285.7*temp/(1+z)/57.3
        return dist
    
    def load_BCG_cat(self):
        self.cat1 = dio.readfile(self.input_path)
        self.cat1['N_r500'] = 0

    def load_background_galaxy_cat(self):
        backg_gal_path = '../data/backg_gal.fits'
        self.cat2 = dio.readfile(backg_gal_path)

    def save_processed_file(self):
        dio.savefile(self.cat1, self.output_path)
    
    # 计算不同范围下成员星系数目
    def get_num_mem(self):
        self.load_BCG_cat()
        self.load_background_galaxy_cat()
        n = 0
        for gind in self.cat1.index:
            # obtain coordinate data of the chosen galaxy cluster
            gdata = self.cat1.loc[gind,:]
            gra = gdata.ra
            gdec = gdata.dec
            gz = gdata.z
            # object radial distance
            rad_dist = self.dis_z(gz)
            # narrow the range of background catalog
            ind = (self.cat2.ra < (gra+0.5)) & (self.cat2.ra > (gra-0.5))
            ind = ind & ((self.cat2.dec < (gdec+0.5)) & (self.cat2.dec > (gdec-0.5)))
            lst_near = self.cat2[ind]
            # difference between each coordinate
            delta_ra = lst_near.ra - gra
            delta_dec = lst_near.dec - gdec
            delta_z = np.abs(lst_near.z - gz)
            # angle scale difference
            ang_diff = np.sqrt(np.abs(delta_ra)**2*np.cos(gdec*self.ang)**2+np.abs(delta_dec)**2)
            dist_diff = ang_diff*rad_dist
            # redshift difference criterion
            z_criterion_phot = 0.04*(1+gz) if gz <= 0.45 else (0.248*gz - 0.0536)
            z_criterion_spec = gdata['specz_slice']
            # separate galaxies based on whether they have spectroscopic redshift
            cond_phot = (dist_diff < gdata['pred_r500']) & (delta_z < z_criterion_phot) & (lst_near.is_spec == 0)
            cond_spec = (dist_diff < gdata['pred_r500']) & (delta_z < z_criterion_spec) & (lst_near.is_spec == 1)
            cond = cond_phot | cond_spec

            if np.sum(cond)>0:
                self.cat1.loc[gind,'N_r500'] = np.sum(cond)
                # calculate mean/median redshift of all member galaxies
                mem = lst_near[cond]
                self.cat1.loc[gind,'z_mem_mean'] = np.mean(mem['z'].values)
                self.cat1.loc[gind,'z_mem_median'] = np.median(mem['z'].values)
                self.cat1.loc[gind,'z_specmem_mean'] = -1
                self.cat1.loc[gind,'z_specmem_median'] = -1
                # calculate mean/median redshift of spec member galaxies
                ind = mem.is_spec>0
                if sum(ind)>0:
                    mem_spec = mem[ind]
                    self.cat1.loc[gind,'z_specmem_mean'] = np.mean(mem_spec['z'].values)
                    self.cat1.loc[gind,'z_specmem_median'] = np.median(mem_spec['z'].values)

            n += 1
            if n%10000==0:
                print('%d / %d.'%(n,len(self.cat1)))
            
        self.cat1 = self.cat1.sort_values('ind_ref',ascending=True)
        self.save_processed_file()
        return self.cat1
    
    
    
if __name__ == '__main__':
    path = '../output/BCG_cand.fits'
    cand = MemberGalaxy(path,path).get_num_mem()
