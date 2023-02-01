
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
from torchvision import models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
    
class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        tmp_x1 = self.conv1(x)
        tmp_x2 = self.relu(tmp_x1)
        tmp_x3 = self.conv2(tmp_x2)
        tmp_x4 = self.relu(x + tmp_x3)
        return tmp_x4
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        
        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()

        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.residual1 = residual_block(64, 64)
        self.cnn2 = nn.Sequential(  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                    nn.MaxPool2d(2, 2, 0, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),)
        self.residual2 = residual_block(128, 128)
        
        self.cnn3 =  nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                    #nn.MaxPool2d(2, 2, 0, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                )
        self.residual3 = residual_block(128, 256)
        self.cnn4 =  nn.Sequential( nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                    nn.MaxPool2d(2, 2, 0, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                )
        self.residual4 = residual_block(256, 256)
        self.cnn5 =  nn.Sequential( nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.MaxPool2d(2, 2, 0, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                )
        self.residual5 = residual_block(256, 256)
        self.fc_1 = nn.Sequential(nn.Linear(256, 128))
        self.fc_relu = nn.ReLU()
        self.fc_2 = nn.Sequential(nn.Linear(128, 10))
        self.dropout_1 = nn.Dropout(0.5)
        # self.dropout_2 = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        
    def forward(self,x):
        x = self.stem_conv(x)
        x = self.residual1(x)
        x = self.cnn2(x)
        x = self.residual2(x)
        # x = self.dropout_1(x)
        x = self.cnn3(x)
        x = self.residual3(x)
        x = self.dropout_1(x)
        x = self.cnn4(x)
        x = self.residual4(x)
        x = self.dropout_1(x)
        x = self.cnn5(x)
        x = self.residual5(x)
        x = self.dropout_1(x) # good1 no this 
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc_1(x)
        x = self.fc_relu(x)
        x = self.dropout_1(x) # good1 dropout_2
        x = self.fc_2(x)
        return x
