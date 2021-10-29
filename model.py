#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:44:25 2021

@author: maclab
"""


"""
Name       : model.py
Function   : The EDM-subgenre-classfication model and weight initialization
Model      : Short-chunk cnn + Resnet       ==> "ShortChunkCNN_Res"
             Short-chunk cnn + Late-fusion  ==> "Joint_ShortChunkCNN_Res_late"
             Short-chunk cnn + early-fusion ==> "Joint_ShortChunkCNN_Res_early"
           
Parameters : n_channel ==> Number of input mel-spectrogram channels
             n_class   ==> Number of EDM-subgenres
             n_frame   ==> The length of Input frame
             
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from modules import Res_2d

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Joint_ShortChunkCNN_Res(nn.Module):
    
    def __init__(self,
                n_channels=128,
                n_class=30):
        super(Joint_ShortChunkCNN_Res, self).__init__()
        
        self.name     = "late"
        # CNN
        self.layer1   = Res_2d(1, n_channels, stride=2)
        self.layer2   = Res_2d(n_channels, n_channels, stride=2)
        self.layer3   = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4   = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5   = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6   = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7   = Res_2d(n_channels*2, n_channels*4, stride=2)
        
        # Dense
        self.dense1   = nn.Linear(n_channels*4+384, n_channels*4)
        self.bn       = nn.BatchNorm1d(n_channels*4)
        self.dense2   = nn.Linear(n_channels*4, n_class)
        self.dropout  = nn.Dropout(0.5)
        self.relu     = nn.ReLU()
        
        # Late-fusion cnn
        self.conv1d_1 = nn.Conv1d(50, 50, kernel_size=3, stride=2)
        self.conv1d_2 = nn.Conv1d(50, 50, kernel_size=3, stride=3)
        self.conv1d_3 = nn.Conv1d(50, 50, kernel_size=5, stride=3)
        self.conv1d_4 = nn.Conv1d(50, 50, kernel_size=5, stride=5)
        self.conv1d_w = nn.Conv1d(1, 1, kernel_size=3, stride=2)
        self.bn_1     = nn.BatchNorm1d(50)
        self.avgpool  = nn.AvgPool1d(50)
        self.maxpool  = nn.MaxPool1d(3, stride=3)
        self.bn_w     = nn.BatchNorm1d(1)

    def forward(self, x, y, z):
        # Late-fusion cnn
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
        y = torch.reshape(y, (y.shape[0], y.shape[2], y.shape[1]))
        x1 = self.conv1d_1(x)
        x1 = self.bn_1(x)
        x1 = self.relu(x)
        x2 = self.conv1d_2(x)
        x2 = self.bn_1(x)
        x2 = self.relu(x)
        x3 = self.conv1d_3(x)
        x3 = self.bn_1(x)
        x3 = self.relu(x)
        x4 = self.conv1d_4(x)
        x4 = self.bn_1(x)
        x4 = self.relu(x)
        x = torch.cat((x1, x2, x3, x4), dim=2)
        y1 = self.conv1d_1(y)
        y1 = self.bn_1(y)
        y1 = self.relu(y)
        y2 = self.conv1d_2(y)
        y2 = self.bn_1(y)
        y2 = self.relu(y)
        y3 = self.conv1d_3(y)
        y3 = self.bn_1(y)
        y3 = self.relu(y)
        y4 = self.conv1d_4(y)
        y4 = self.bn_1(y)
        y4 = self.relu(y)
        y = torch.cat((y1, y2, y3, y4), dim=2)
        w = torch.cat((x, y), dim=2)
        w = torch.reshape(w, (w.shape[0], w.shape[2], w.shape[1]))
        w = self.avgpool(w)
        w = torch.reshape(w, (w.shape[0], w.shape[2], w.shape[1]))
        w = self.conv1d_w(w)
        w = self.bn_w(w)
        w = self.relu(w)
        w = self.maxpool(w)
        w = torch.flatten(w, 1)
        
        # CNN
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        z = z.squeeze(2)

        # Global Max Pooling
        if z.size(-1) != 1:
            z = nn.MaxPool1d(z.size(-1))(z)
        z = z.squeeze(2)

        # Dense
        z = torch.cat((z, w), dim=1)
        z = self.dense1(z)
        z = self.bn(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.dense2(z)
        z = nn.Softmax()(z)

        return z

class Joint_ShortChunkCNN_Res_early(nn.Module):
    
    def __init__(self,
                n_channels=128,
                n_class=30,
                n_frame = 50):
        super(Joint_ShortChunkCNN_Res_early, self).__init__()
        
        self.name     = "early"
        # CNN
        self.layer1   = Res_2d(1, n_channels, stride=2)
        self.layer2   = Res_2d(n_channels, n_channels, stride=2)
        self.layer3   = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4   = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5   = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6   = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7   = Res_2d(n_channels*2, n_channels*4, stride=2)
        
        # Dense
        self.dense1   = nn.Linear(n_channels*4+384, n_channels*4)
        self.bn       = nn.BatchNorm1d(n_channels*4)
        self.dense2   = nn.Linear(n_channels*4, n_class)
        self.dropout  = nn.Dropout(0.5)
        self.relu     = nn.ReLU()
        
        # Late-fusion cnn
        self.conv1d_1 = nn.Conv1d(n_frame, n_frame, kernel_size=3, stride=2)
        self.conv1d_2 = nn.Conv1d(n_frame, n_frame, kernel_size=3, stride=3)
        self.conv1d_3 = nn.Conv1d(n_frame, n_frame, kernel_size=5, stride=3)
        self.conv1d_4 = nn.Conv1d(n_frame, n_frame, kernel_size=5, stride=5)
        self.conv1d_w = nn.Conv1d(1, 1, kernel_size=3, stride=2)
        self.bn_1     = nn.BatchNorm1d(n_frame)
        self.avgpool  = nn.AvgPool1d(n_frame)
        self.maxpool  = nn.MaxPool1d(3, stride=3)
        self.bn_w     = nn.BatchNorm1d(1)
        self.rnn      = nn.GRU(n_frame,1,1)
#       self.fc       = nn.Linear(128, 64)

    def forward(self, x, y, z):
        # Late-fusion cnn
        x  = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
        y  = torch.reshape(y, (y.shape[0], y.shape[2], y.shape[1]))
        x  = torch.cat((x, y), dim=2)
        x1 = self.conv1d_1(x)
        x1 = self.bn_1(x)
        x1 = self.relu(x)
        x2 = self.conv1d_2(x)
        x2 = self.bn_1(x)
        x2 = self.relu(x)
        x3 = self.conv1d_3(x)
        x3 = self.bn_1(x)
        x3 = self.relu(x)
        x4 = self.conv1d_4(x)
        x4 = self.bn_1(x)
        x4 = self.relu(x)
        w  = torch.cat((x1, x2, x3, x4), dim=2)
        w = torch.reshape(w, (w.shape[0], w.shape[2], w.shape[1]))
        w = self.avgpool(w)
        w = torch.reshape(w, (w.shape[0], w.shape[2], w.shape[1]))
        w = self.conv1d_w(w)
        w = self.bn_w(w)
        w = self.relu(w)
        w = self.maxpool(w)
        w = torch.flatten(w, 1)
        
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        z = z.squeeze(2)

        # Global Max Pooling
        if z.size(-1) != 1:
            z = nn.MaxPool1d(z.size(-1))(z)
        z = z.squeeze(2)

        # Dense
        z = torch.cat((z, w), dim=1)
        z = self.dense1(z)
        z = self.bn(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.dense2(z)
        z = nn.Softmax()(z)

        return z
    
class ShortChunkCNN_Res(nn.Module):
    '''
    Short-chunk CNN architecture with residual connections.
    '''
    def __init__(self,
                n_channels=128,
                n_class=30):
        super(ShortChunkCNN_Res, self).__init__()
        
        self.name     = "SCcnn"
        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)
        
        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)
        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Softmax()(x)

        return x
    
