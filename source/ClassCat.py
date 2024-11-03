from typing import Any
import h5py
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import ResNet_hyperparams as rh


class Cluster_dataset(Dataset):
    def __init__(self,file_paths,mode='inference'):
        self.file_paths = file_paths
        self.mode = mode

    def gsm_redshift_correction(self, image, z):
        Omega_m = 0.3
        Omega_Lambda = 0.7
        Ez = (Omega_Lambda + Omega_m*(1 + z)**3)**0.5
        return image * Ez
    
    def sub_background(self, image, back):
        return image - back

    def __len__(self):
        total_length = 0
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                total_length += len(file['x'])
        return total_length

    def __getitem__(self, index):
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                image = torch.tensor(file['x'][index,0,:,:], dtype=torch.float32)
                image = torch.unsqueeze(image, dim=0)

                z = torch.tensor(file['z'][index], dtype=torch.float32)
                backg = torch.tensor(file['backg'][index], dtype=torch.float32)

                # Apply processing based on the specified order
                image = self.sub_background(image, backg)
                image = self.gsm_redshift_correction(image, z)

                if self.mode=='train':
                    label = torch.tensor(file['y'][index], dtype=torch.float32)
                    label = torch.unsqueeze(label, dim=-1)
                    return image, label
                return image



class ResNet(nn.Module):
    """ 
    加载ResNet模型注意: \\
    加载模型时,同时需要修改class,模型的类必须与训练一致 \\
    加载出来的模型就是对应train_num的模型
    """
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = models.resnet34(pretrained=True)
        # 冻结所有模型参数
        # for param in self.net.parameters():
        #     param.requires_grad = False
    
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1, bias=False)
        self.net.fc1 = nn.Sequential(
                            nn.Linear(self.net.fc.out_features, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1))
    
    def forward(self, x):
        y = self.net.conv1(x)
        y = self.net.bn1(y)
        y = self.net.relu(y)
        y = self.net.maxpool(y)
        y = self.net.layer1(y)
        y = self.net.layer2(y)
        y = self.net.layer3(y)
        y = self.net.layer4(y)
        y = self.net.avgpool(y)
        y = torch.squeeze(y)
        y = self.net.fc(y)
        y = self.net.fc1(y)
        return y



class EarlyStoppiong:
    """ 
    Early stopping机制是一种正则化手段，用于避免模型过拟合。\\
    Early stopping会跟踪验证损失(val_loss)，若连续几个epoch停止下降，则停止训练 \\
    每次val_loss减少时，该类都会保存模型的一个检查点。 
    """
    def __init__(self, patience, path, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print('EarlyStopping counter: %d / %d'%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ val_loss下降时保存模型至checkpoint.pkl """
        if self.verbose:
            print('Validation loss decreased: %.3f --> %.3f   Saving model ...'%(self.val_loss_min, val_loss))
        torch.save(model, self.path)
        self.val_loss_min = val_loss
        


class Logger(object):
    """ 保存控制台输出 """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass



def print_model_hyperparams():
    """ print hyperparameters model used """
    import importlib
    # 导入 Hyperparams.py 文件
    config_module = importlib.import_module('ResNet_hyperparams')
    # 获取模块中的全局变量字典
    global_variables = config_module.__dict__
    # 遍历字典按照变量顺序打印
    for variable_name, variable_value in global_variables.items():
        if not variable_name.startswith("__"):  # 排除内置变量
            print(f"{variable_name}: {variable_value}")


def plot_loss_curve(epoch,train_losses,test_losses,logdir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 2), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(logdir + 'loss_curve_%03d.png' % (rh.train_num))
    plt.close()


def plot_resnet_train_result(pred_richness_train, richness_train,
                             pred_richness_test, richness_test, logdir):
    import matplotlib.pyplot as plt
    import functions as f

    # 计算bias和弥散
    bias_train = f.cal_bias(pred_richness_train, richness_train)
    dispersion_train, _ = f.cal_dispersion(pred_richness_train, richness_train)
    bias_test = f.cal_bias(pred_richness_test, richness_test)
    dispersion_test, _ = f.cal_dispersion(pred_richness_test, richness_test)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    plt.scatter(richness_train, pred_richness_train, marker='.', s=8, color='k', alpha=0.4, zorder=2)
    plt.xlabel('Label')
    plt.ylabel('Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.text(0.05, 0.95, 'bias: %.3f \ndispersion: %.3f' % (bias_train, dispersion_train),
             transform=ax.transAxes, verticalalignment="top", horizontalalignment="left")
    # plt.gca().set_aspect('equal', adjustable='box')  # Maintain square aspect ratio

    ax = fig.add_subplot(122)
    plt.scatter(richness_test, pred_richness_test, marker='.', s=8, color='k', alpha=0.4, zorder=2)
    plt.xlabel('Label')
    plt.xscale('log')
    plt.yscale('log')
    plt.text(0.05, 0.95, s='bias: %.3f \ndispersion: %.3f' % (bias_test, dispersion_test),
             transform=ax.transAxes, verticalalignment="top", horizontalalignment="left")
    # plt.gca().set_aspect('equal', adjustable='box')  # Maintain square aspect ratio
    
    plt.savefig(logdir + 'train_test_result_%03d.png' % (rh.train_num))
