#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:10:57 2021

@author: maclab
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import librosa
import warnings
warnings.filterwarnings("ignore")

##### INPUT DATA PREPROCESSING #####

def joint_data(data_type = 0, dataset_ratio = 0.1):

    genre_list = os.listdir("./data/tempogram/")
    edm_melspec_1 = []
    edm_melspec_2 = []
    edm_melspec_3 = []
    for genre in genre_list:
        locals()['data_'+f'{genre}'+'t'] = []
        locals()['data_'+f'{genre}'+'t'].append(f'{genre}')
        locals()['data_'+f'{genre}'+'f'] = []
        locals()['data_'+f'{genre}'+'f'].append(f'{genre}')
        locals()['data_'+f'{genre}'] = []
        locals()['data_'+f'{genre}'].append(f'{genre}')
        locals()['temp_dir_'+f'{genre}'+'t'] = []
        locals()['temp_dir_'+f'{genre}'+'f'] = []
        locals()['temp_dir_'+f'{genre}'] = []
        temp_list = os.listdir(f'./data/melspec/{genre}/')
        for i in tqdm(range(int(len(temp_list)*dataset_ratio))):
            t  = np.load(f'./data/melspec/{genre}/'+temp_list[i][:-3]+'npy')
            t1 = np.load(f'./data/tempogram/{genre}/'+temp_list[i][:-4]+'.mp3'+'.npy')
            t2 = np.load(f'./data/tempogram/{genre}/'+'F_'+temp_list[i][:-4]+'.mp3'+'.npy')
            t  = np.array(t, dtype='float32')
            t1  = np.array(t1, dtype='float32')
            t2  = np.array(t2, dtype='float32')
            t  = (t  - np.mean(t))/np.std(t)
            t1 = (t1 - np.mean(t1))/np.std(t1)
            t2 = (t2 - np.mean(t2))/np.std(t2)
            if (not np.any(np.isnan(t))) and (not np.any(np.isnan(t1))) and (not np.any(np.isnan(t2))) and (t.shape[0] == 5168):
                locals()['temp_dir_'+f'{genre}'+'t'].append(temp_list[i])
                locals()['temp_dir_'+f'{genre}'+'f'].append(temp_list[i])
                locals()['temp_dir_'+f'{genre}'].append(temp_list[i])
        print("deal with {} ...".format(genre))
        for k in tqdm(range(len(locals()['temp_dir_'+f'{genre}'+'t']))):
            temp_1 = []
            t = np.load(f'./data/tempogram/{genre}/'+locals()['temp_dir_'+f'{genre}'+'t'][k][:-4]+'.mp3'+'.npy')
            t = np.array(t, dtype='float32')
            t = (t - np.mean(t))/np.std(t)
            if not np.any(np.isnan(t)):
                temp_1.append(locals()['temp_dir_'+f'{genre}'+'t'][k])
                temp_1.append(t)
                locals()['data_'+f'{genre}'+'t'].append(temp_1)
        for k in tqdm(range(len(locals()['temp_dir_'+f'{genre}'+'f']))):
            temp_2 = []
            t = np.load(f'./data/tempogram/{genre}/'+'F_'+locals()['temp_dir_'+f'{genre}'+'f'][k][:-4]+'.mp3'+'.npy')
            t = np.array(t, dtype='float32')
            t = (t - np.mean(t))/np.std(t)
            if not np.any(np.isnan(t)):
                temp_2.append(locals()['temp_dir_'+f'{genre}'+'f'][k])
                temp_2.append(t)
                locals()['data_'+f'{genre}'+'f'].append(temp_2)
        for k in tqdm(range(len(locals()['temp_dir_'+f'{genre}']))):
            temp_3 = []
            t = np.load(f'./data/melspec/{genre}/'+locals()['temp_dir_'+f'{genre}'][k][:-4]+'.npy')
            t = np.array(t, dtype='float32')
            t = t.transpose()
            t = (t - np.mean(t))/np.std(t)
            if not np.any(np.isnan(t)):
                temp_3.append(locals()['temp_dir_'+f'{genre}'][k])
                temp_3.append(t)
                locals()['data_'+f'{genre}'].append(temp_3)
        edm_melspec_1.append(locals()['data_'+f'{genre}'+'t'])
        edm_melspec_2.append(locals()['data_'+f'{genre}'+'f'])
        edm_melspec_3.append(locals()['data_'+f'{genre}'])
    if data_type == 0:
        SC_cnn_domain = 50
    else :
        SC_cnn_domain = 200
    X_train_1 = []
    Y_train_1 = []
    Z_train_1 = []
    X_test_1 = []
    Y_test_1 = []
    Z_test_1 = []
    X_val_1 = []
    Y_val_1 = []
    Z_val_1 = []
    for i in tqdm(range(len(edm_melspec_1))):
        if type(edm_melspec_1[i][0]) == str:
            edm_melspec_1[i].remove(edm_melspec_1[i][0])
        temp_f_train_1, temp_f_test_1 = train_test_split(edm_melspec_1[i], test_size = 0.2, random_state = 10, shuffle=False)
        temp_f_val_1, temp_f_test_1 = train_test_split(temp_f_test_1, test_size = 0.5, random_state = 10, shuffle=False)
        for j in range(len(temp_f_train_1)):
            for k in range(int(temp_f_train_1[j][1].shape[1]/SC_cnn_domain)):
                X_train_1.append(temp_f_train_1[j][1][:,(0+SC_cnn_domain*k):(SC_cnn_domain+SC_cnn_domain*k)])
                Y_train_1.append(int(np.argwhere(np.array(genre_list)==genre_list[i])))
                Z_train_1.append(temp_f_train_1[j][0])
        for l in range(len(temp_f_test_1)):
            for k in range(int(temp_f_train_1[l][1].shape[1]/SC_cnn_domain)):
                X_test_1.append(temp_f_test_1[l][1][:,(0+SC_cnn_domain*k):(SC_cnn_domain+SC_cnn_domain*k)])
                Y_test_1.append(int(np.argwhere(np.array(genre_list)==genre_list[i])))
                Z_test_1.append(temp_f_test_1[l][0])
        for l in range(len(temp_f_val_1)):
            for k in range(int(temp_f_val_1[l][1].shape[1]/SC_cnn_domain)):
                X_val_1.append(temp_f_val_1[l][1][:,(0+SC_cnn_domain*k):(SC_cnn_domain+SC_cnn_domain*k)])
                Y_val_1.append(int(np.argwhere(np.array(genre_list)==genre_list[i])))
                Z_val_1.append(temp_f_val_1[l][0])
            
    X_train_1 = np.array(X_train_1)
    X_test_1 = np.array(X_test_1)
    X_val_1 = np.array(X_val_1)
    Y_train_1 = np.array(Y_train_1)
    Y_test_1 = np.array(Y_test_1)
    Y_val_1 = np.array(Y_val_1)
    
    if data_type == 0:
        SC_cnn_domain = 50
    else :
        SC_cnn_domain = 200
    X_train_2 = []
    X_test_2 = []
    X_val_2 = []
    for i in tqdm(range(len(edm_melspec_2))):
        if type(edm_melspec_2[i][0]) == str:
            edm_melspec_2[i].remove(edm_melspec_2[i][0])
        temp_f_train_2, temp_f_test_2 = train_test_split(edm_melspec_2[i], test_size = 0.2, random_state = 10, shuffle=False)
        temp_f_val_2, temp_f_test_2 = train_test_split(temp_f_test_2, test_size = 0.5, random_state = 10, shuffle=False)
        for j in range(len(temp_f_train_2)):
            for k in range(int(temp_f_train_2[j][1].shape[1]/SC_cnn_domain)):
                X_train_2.append(temp_f_train_2[j][1][:,(0+SC_cnn_domain*k):(SC_cnn_domain+SC_cnn_domain*k)])
        for l in range(len(temp_f_test_2)):
            for m in range(int(temp_f_test_2[l][1].shape[1]/SC_cnn_domain)):
                X_test_2.append(temp_f_test_2[l][1][:,(0+SC_cnn_domain*m):(SC_cnn_domain+SC_cnn_domain*m)])
        for l in range(len(temp_f_val_2)):
            for m in range(int(temp_f_val_2[l][1].shape[1]/SC_cnn_domain)):
                X_val_2.append(temp_f_val_2[l][1][:,(0+SC_cnn_domain*m):(SC_cnn_domain+SC_cnn_domain*m)])

    X_train_2 = np.array(X_train_2)
    X_test_2 = np.array(X_test_2)
    X_val_2 = np.array(X_val_2)

    SC_cnn_domain = 200    
    X_train_3 = []
    X_test_3  = []
    X_val_3 = []
    
    for i in tqdm(range(len(edm_melspec_3))):
        if type(edm_melspec_3[i][0]) == str:
            edm_melspec_3[i].remove(edm_melspec_3[i][0])
        temp_f_train, temp_f_test = train_test_split(edm_melspec_3[i], test_size = 0.2, random_state = 10, shuffle = False)
        temp_f_val, temp_f_test = train_test_split(temp_f_test, test_size = 0.5, random_state = 10, shuffle = False)
        for j in range(len(temp_f_train)):
            for k in range(int(temp_f_train[j][1].shape[1]/SC_cnn_domain)):
                X_train_3.append(temp_f_train[j][1][:,(0+SC_cnn_domain*k):(SC_cnn_domain+SC_cnn_domain*k)])
        for l in range(len(temp_f_test)):
            for m in range(int(temp_f_test[l][1].shape[1]/SC_cnn_domain)):
                X_test_3.append(temp_f_test[l][1][:,(0+SC_cnn_domain*m):(SC_cnn_domain+SC_cnn_domain*m)])
        for l in range(len(temp_f_val)):
            for m in range(int(temp_f_val[l][1].shape[1]/SC_cnn_domain)):
                X_val_3.append(temp_f_val[l][1][:,(0+SC_cnn_domain*m):(SC_cnn_domain+SC_cnn_domain*m)])
    
        
    X_train_3 = np.array(X_train_3)
    X_test_3  = np.array(X_test_3)
    X_val_3 = np.array(X_val_3)
            
    featuresTrain_1 = torch.from_numpy(X_train_1)
    targetsTrain_1  = torch.from_numpy(Y_train_1).type(torch.LongTensor) # data type is long
            
    featuresTest_1  = torch.from_numpy(X_test_1)
    targetsTest_1   = torch.from_numpy(Y_test_1).type(torch.LongTensor) # data type is long
            
    featuresVal_1   = torch.from_numpy(X_val_1)
    targetsVal_1    = torch.from_numpy(Y_val_1).type(torch.LongTensor) # data type is long
            
    train_1         = torch.utils.data.TensorDataset(featuresTrain_1, targetsTrain_1)
    test_1          = torch.utils.data.TensorDataset(featuresTest_1, targetsTest_1)
    val_1           = torch.utils.data.TensorDataset(featuresVal_1, targetsVal_1)
                        
    featuresTrain_2 = torch.from_numpy(X_train_2)        
    featuresTest_2  = torch.from_numpy(X_test_2)            
    featuresVal_2   = torch.from_numpy(X_val_2)
            
    train_2         = torch.utils.data.TensorDataset(featuresTrain_2, targetsTrain_1)
    test_2          = torch.utils.data.TensorDataset(featuresTest_2, targetsTest_1)
    val_2           = torch.utils.data.TensorDataset(featuresVal_2, targetsVal_1)
          
    featuresTrain_3 = torch.from_numpy(X_train_3)            
    featuresTest_3  = torch.from_numpy(X_test_3)           
    featuresVal_3   = torch.from_numpy(X_val_3)

    train_3         = torch.utils.data.TensorDataset(featuresTrain_3, targetsTrain_1)
    test_3          = torch.utils.data.TensorDataset(featuresTest_3, targetsTest_1)
    val_3           = torch.utils.data.TensorDataset(featuresVal_3, targetsVal_1)
    
    return train_1, test_1, val_1, train_2, test_2, val_2, train_3, test_3, val_3, Z_test_1, targetsTest_1

def audio_to_mel_and_temp(audio):
    
    y, sr       = librosa.load(audio, sr=22050)
    S           = librosa.feature.melspectrogram(y=y, sr=22050)
    oenv        = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)
    a_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=22050,
                                      hop_length=512)
    f_tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=22050,
                                              hop_length=512)
    return S, a_tempogram, f_tempogram

def save_feature(data_folder = "./data/audio/"):

    audio_list  = os.listdir(data_folder)
    for i in tqdm(range(len(audio_list))):
        S, a, f = audio_to_mel_and_temp("./data/audio/" + audio_list[i])
        np.save("./data/mel-spectrogram/mel_spec_{}".format(audio_list[i]), S)
        np.save("./data/auto-tempogram/auto_tempogram_{}".format(audio_list[i]), a)
        np.save("./data/fourier-tempogram/fourier_tempogram_{}".format(audio_list[i]), f)
    print("done!")

def set_the_dataset_using(dataset = "./data/"):

    data_t     = []
    data_f     = []
    data_      = []
    temp_dir_t = []
    temp_dir_f = []
    temp_dir_  = []
    
    temp_list  = os.listdir(f'./data/audio/')
    
    for i in tqdm(range(int(len(temp_list)))):
        t  = np.load(f'./data/auto-tempogram/auto_tempogram_'+temp_list[i]+'.npy')
        t1 = np.load(f'./data/fourier-tempogram/fourier_tempogram_'+temp_list[i]+'.npy')
        t2 = np.load(f'./data/mel-spectrogram/'+'mel_spec_'+temp_list[i]+'.npy')
        t  = np.array(t, dtype='float32')
        t  = np.array(t1, dtype='float32')
        t  = np.array(t2, dtype='float32')
        t  = (t  - np.mean(t))/np.std(t)
        t1 = (t1 - np.mean(t1))/np.std(t1)
        t2 = (t2 - np.mean(t2))/np.std(t2)
        if (not np.any(np.isnan(t))) and (not np.any(np.isnan(t1))) and (not np.any(np.isnan(t2))):
            temp_dir_t.append(temp_list[i])
            temp_dir_f.append(temp_list[i])
            temp_dir_.append(temp_list[i])
    for k in tqdm(range(len(temp_dir_t))):
        temp_1 = []
        t = np.load(f'./data/auto-tempogram/auto_tempogram_'+temp_dir_t[k]+'.npy')
        t = np.array(t, dtype='float32')
        t = t[:,1292:1292*2]
        t = (t - np.mean(t))/np.std(t)
        if not np.any(np.isnan(t)):
            temp_1.append(temp_dir_t[k])
            temp_1.append(t)
            data_t.append(temp_1)
    for k in tqdm(range(len(temp_dir_f))):
        temp_2 = []
        t = np.load(f'./data/fourier-tempogram/fourier_tempogram_'+temp_dir_f[k]+'.npy')
        t = np.array(t, dtype='float32')
        t = t[:,1292:1292*2]
        t = (t - np.mean(t))/np.std(t)
        if not np.any(np.isnan(t)):
            temp_2.append(temp_dir_f[k])
            temp_2.append(t)
            data_f.append(temp_2)
    for k in tqdm(range(len(temp_dir_))):
        temp_3 = []
        t = np.load(f'./data/mel-spectrogram/mel_spec_'+temp_dir_[k]+'.npy')
        t = np.array(t, dtype='float32')
        t = t[:,0:5168]
        t = (t - np.mean(t))/np.std(t)
        if not np.any(np.isnan(t)):
            temp_3.append(temp_dir_[k])
            temp_3.append(t)
            data_.append(temp_3)

    a_tempo         = 50
    f_tempo         = 50
    SC_cnn_domain_m = 200
    X_test_t  = []
    Z_test_t  = []
    X_test_f  = []
    X_test_m  = []

    for l in tqdm(range(len(data_t))):
        for k in range(int(data_t[l][1].shape[1]/a_tempo)):
            X_test_t.append(data_t[l][1][:,(0 + a_tempo*k):(a_tempo + a_tempo*k)])
            X_test_f.append(data_f[l][1][:,(0 + f_tempo*k):(f_tempo + f_tempo*k)])
            X_test_m.append(data_[l][1][:,(0 + SC_cnn_domain_m*k):(SC_cnn_domain_m + SC_cnn_domain_m*k)])
            Z_test_t.append(data_t[l][0])
            
    X_test_t = np.array(X_test_t)
    X_test_f = np.array(X_test_f)
    X_test_m = np.array(X_test_m)

    featuresTest_t  = torch.from_numpy(X_test_t)      
    featuresTest_f  = torch.from_numpy(X_test_f)            
    featuresTest_m  = torch.from_numpy(X_test_m)
            
    feature_t       = torch.utils.data.TensorDataset(featuresTest_t)
    feature_f       = torch.utils.data.TensorDataset(featuresTest_f)
    feature_m       = torch.utils.data.TensorDataset(featuresTest_m)
    
    return feature_t, feature_f, feature_m, Z_test_t
    