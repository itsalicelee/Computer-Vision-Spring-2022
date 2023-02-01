
import torch
import os

import parser
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from myModels import  myLeNet, myResnet, weights_init
from myDatasets import  get_cifar10_train_val_set
from tool import train, fixed_seed
#from torchsummary import summary 


def train_interface():
    args = parser.arg_parse()
    """ input argument """

    data_root = args.data_root
    model_type = args.model_type
    num_out = args.num_out
    num_epoch = args.epochs
    split_ratio = args.split_ratio
    seed = args.seed
    
    # fixed random seed
    fixed_seed(seed)
    
    
    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    os.makedirs(os.path.join('./curve', model_type), exist_ok=True)
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '.log')
    save_path = os.path.join('./save_dir', model_type)
    curve_path = os.path.join('./curve', model_type)
    
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(log_path, 'w'):
        pass
    
    
    
    """ training hyperparameter """
    lr = args.lr
    batch_size = args.batch_size
    milestones = args.milestones
    
    #TODO: 
    ## Modify here if you want to change your model ##
    # model = myLeNet(num_out=num_out)
    model = myResnet()
    # model = resnet()
    model.apply(weights_init)
    # print model's architecture
    print(model)
    

    



    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay) # good with Adam 1e-5
    # scheduler = None
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.6) # good 
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    #print(summary(model, (3, 32, 32)))
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path, curve_path=curve_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_set=train_set)

    
if __name__ == '__main__':
    train_interface()




    