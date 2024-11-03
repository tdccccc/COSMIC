# Hyperparameters of ResNet training
# Modify params below
# Then run 'nohup python3 ResNet_trian_sp.py' in command line
random_seed = 2024  # int

GPU_ID = '1'  # str,
train_num = 10  # int
mode = 'train' # str, ['train','inference']

# hyperparameters of training
model = 'ResNet34'
pretrained = True
batch_size = 32  # int
epoch = 10  # int

init_lr = 0.01  # float
lr_decay_factor = 0.1  # float
lr_decay_alpha = 1 # float
optimizer = 'SGD'  # str, ['Adam','SGD','ASGD','AdamW']
optimizer_weight_decay = 0.01  # float
lossfunction = 'MSELoss'  # str, ['MSELoss', 'SmoothL1Loss',]
activation_function = 'ReLU'  # str, ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU',]

train_file = ['../data/trainset_seed2024_fixeddv2500_sigma0.05_1mpc.h5']
test_file = ['../data/testset_seed2024_fixeddv2500_sigma0.05_1mpc.h5']
