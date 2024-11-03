# Predict richness using trained ResNet model
import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader

import ResNet_hyperparams as rh
from ClassCat import Cluster_dataset, ResNet
import dataio as dio


def mass_est(ResNet_input_file, batch_size=128, num_workers=20):
    # 加载模型
    ResNet_model = torch.load('../model/ResNet_best_model_008.pkl')
    ResNet_model.eval()
    ResNet_model = ResNet_model.cuda()

    # 加载数据集
    dataset = Cluster_dataset(file_paths=ResNet_input_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    num = len(dataset)
    res = np.zeros(num)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            imgs = batch.cuda(non_blocking=True)
            predictions = ResNet_model(imgs)
            predictions = torch.squeeze(predictions)
            current_batch_size = len(predictions)
            res[i*batch_size:i*batch_size+current_batch_size] = predictions.cpu().numpy()
            
            if i % (10000 // batch_size) == 0:
                print(f'{i * batch_size}/{num} done...')
    return res



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定GPU 
    rh.train_num = 8
    rh.mode = 'inference'

    ResNet_input_file = ['../output/ResNet_input.h5']
    res = mass_est(ResNet_input_file=ResNet_input_file,
                    batch_size=128, 
                    num_workers=20)

    path = '../output/BCG_cand.fits'
    BCG_cand = dio.readfile(path)

    path = '../output/BCG_cand.fits'
    BCG_cand['pred_richness'] = res
    dio.savefile(BCG_cand, path)