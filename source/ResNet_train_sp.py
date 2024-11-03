# This file used to train a ResNet model
# run 'nohup python3 ResNet_trian_sp.py' in command line
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import ClassCat as cc
import ResNet_hyperparams as rh
import os
os.environ["CUDA_VISIBLE_DEVICES"] = rh.GPU_ID # 指定GPU
os.chdir(os.path.dirname('/home/tiandc/galaxy_cluster/apjs_reply/source/'))
os.getcwd()
# 创建对应train_num的文件夹
logdir = '../model/ResNet/ResNet_train_log_%03d/'%(rh.train_num)
os.makedirs(logdir, exist_ok=True)

# 将控制台的结果输出到.txt文件
sys.stdout = cc.Logger(logdir+'train_log_%03d.txt'%(rh.train_num), sys.stdout)
sys.stderr = cc.Logger(logdir+'err.log_file', sys.stderr) # 输出报错

trainset = cc.Cluster_dataset(file_paths=rh.train_file,mode='train')
testset = cc.Cluster_dataset(file_paths=rh.test_file,mode='train')
print('Number of training set: %d'%len(trainset))

torch.manual_seed(seed=2024)
trainloader = DataLoader(trainset, batch_size=rh.batch_size, shuffle=True, pin_memory=True)
testloader = DataLoader(testset, batch_size=rh.batch_size, shuffle=True, pin_memory=True)

# file path
checkpointpath = logdir+'ResNet_checkpoint_%03d.pkl'%(rh.train_num)
best_modelpath = logdir+'ResNet_best_model_%03d.pkl'%(rh.train_num)
final_modelpath = logdir+'ResNet_final_model_%03d.pkl'%(rh.train_num)


resnet = cc.ResNet()
if torch.cuda.is_available(): # 判断是否支持 CUDA
    resnet = resnet.cuda()  # 放到 GPU 上
torch.set_num_threads(10)

early_stopping = cc.EarlyStoppiong(patience=3, path=checkpointpath)

if rh.lossfunction == 'MSELoss':
    criterian = nn.MSELoss(reduction='mean')
elif rh.lossfunction == 'SmoothL1Loss':
    criterian = nn.SmoothL1Loss(reduction='mean')

cc.print_model_hyperparams()
print('-----------------training------------------')

train_losses = []
test_losses = []

for i in range(rh.epoch):
    since = time.time()

    # learning rate decay
    learning_rate = rh.init_lr * rh.lr_decay_factor ** (i/rh.lr_decay_alpha)
    # set optimiter
    if rh.optimizer == 'SGD':
        optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9)
    elif rh.optimizer == 'ASGD':
        optimizer = optim.ASGD(resnet.parameters(), lr=learning_rate, weight_decay=0.01)
    elif rh.optimizer == 'Adam':
        optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=0.01)
    elif rh.optimizer == 'AdamW':
        optimizer = optim.AdamW(resnet.parameters(), lr=learning_rate, weight_decay=0.01)

    resnet.train()
    running_loss = 0.0
    for (img, label) in trainloader:
        # forward
        # rotate random 90 deg
        rot_k = np.random.choice(4, 1, replace=False)[0]
        img = Variable(torch.rot90(img, k=rot_k, dims=[2, 3])).cuda()
        label = Variable(label).cuda()
        optimizer.zero_grad()
        output = resnet(img)
        loss = criterian(output, label)**0.5
        # backward
        loss.backward()
        optimizer.step()
        # running loss
        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)
    train_losses.append(epoch_loss)

    resnet.eval()
    test_loss = 0.0
    with torch.no_grad():
        for (img, label) in testloader:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            output = resnet(img)
            loss = criterian(output, label)**0.5

            test_loss += loss.item()
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)

    # 学习率调度器步进
    # scheduler.step(test_loss)

    status_string = "[%d/%d]  Train loss: %.3f | Test loss: %.3f | Time: %.1fs"%(
         i+1, rh.epoch, epoch_loss, test_loss, time.time()-since)
    print(status_string)

    # 每epoch绘制曲线
    if (i + 1) % 1 == 0:
        cc.plot_loss_curve(i,train_losses,test_losses,logdir)

    early_stopping(val_loss=test_loss, model=resnet)
    if early_stopping.early_stop:
        print('Early stopping.')
        break

# 保存最后的损失曲线
cc.plot_loss_curve(i,train_losses,test_losses,logdir)

# 保存训练到最后的模型
torch.save(resnet, final_modelpath)

# 保存测试集损失最小的模型
best_resnet = torch.load(checkpointpath)
torch.save(best_resnet, best_modelpath)

#=======================================================
# 获取训练集和测试集的模型输出
#=======================================================
# 加载模型
model = torch.load(best_modelpath)

# 训练集
pred_label_train = np.zeros([len(trainset)])
label_train = np.zeros([len(trainset)])

n = 0
model.eval()
with torch.no_grad():
    for i in range(len(trainset)):
        img, label_train[i] = trainset.__getitem__(i)
        img = torch.unsqueeze(img.cuda(),dim=0)
        pred_label_train[i] = model(img).detach().cpu().numpy()
        n+=1
        if n%10000 == 0: 
                print('trainset: %d / %d done.'%(n, len(trainset)))

# 测试集
pred_label_test = np.zeros([len(testset)])
label_test = np.zeros([len(testset)])

n = 0
model.eval()
with torch.no_grad():
    for i in range(len(testset)):
        img, label_test[i] = testset.__getitem__(i)
        img = torch.unsqueeze(img.cuda(),dim=0)
        pred_label_test[i] = model(img).detach().cpu().numpy()
        n+=1
        if n%10000 == 0:
                print('testset: %d / %d done.'%(n, len(testset)))

# 保存模型结构
print(model)
# 清除占用的CUDA显存
torch.cuda.empty_cache()

# 保存模型输出
resultdir = logdir+'result/'
os.mkdir(resultdir)
np.savetxt(resultdir+'ResNet_%03d_train_prediction.dat'%(rh.train_num), pred_label_train, fmt='%.4f')
np.savetxt(resultdir+'ResNet_%03d_train.dat'%(rh.train_num), label_train, fmt='%.4f')
np.savetxt(resultdir+'ResNet_%03d_test_prediction.dat'%(rh.train_num), pred_label_test, fmt='%.4f')
np.savetxt(resultdir+'ResNet_%03d_test.dat'%(rh.train_num), label_test, fmt='%.4f')

cc.plot_resnet_train_result(pred_label_train,label_train,
                            pred_label_test, label_test, logdir)
