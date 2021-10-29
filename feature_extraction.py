#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:33:56 2021

@author: maclab
"""

import os
import numpy as np
from tqdm import tqdm
import torch
import librosa
import warnings
warnings.filterwarnings("ignore")

def audio_to_mel_and_temp(audio):
    
    y, sr       = librosa.load(audio, sr=22050)
    S           = librosa.feature.melspectrogram(y=y, sr=22050)
    oenv        = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)
    a_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=22050,
                                      hop_length=512)
    f_tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=22050,
                                              hop_length=512)
    return S, a_tempogram, f_tempogram

def save_feature(data_name = "./data/audio/DJ Planet Express - More Than You'd Ever Wanna Know.mp3"):

    for i in tqdm(range(len(data_name))):
        S, a, f = audio_to_mel_and_temp("./data/audio/" + data_name)
        np.save("./data/mel-spectrogram/mel_spec_{}".format(data_name), S)
        np.save("./data/auto-tempogram/auto_tempogram_{}".format(data_name), a)
        np.save("./data/fourier-tempogram/fourier_tempogram_{}".format(data_name), f)
    print("Features are be saved !")

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
