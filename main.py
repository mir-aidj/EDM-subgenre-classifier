#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:05:01 2021

@author: maclab
"""

"""
Name     : main.py
Function : Execute the EDM-subgenre-classification (Just for Using)

This program provide late-fusion pre-trained model, and the 
file name is "joint-model-all-2500.pkl". The pre-trained model 
is in under the folder "model_pkl".

"""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,6"
import sys
import torch
import torch.nn as nn
from model import Joint_ShortChunkCNN_Res
from feature_extraction import set_the_dataset_using, save_feature
from dataset_concat import ConcatDataset
from model_loading_predict import Using_joint_model, song_level_test

# EDM subgenre list with the original order 
if __name__ == "__main__":
    
    classifier_type   = "using" 
    
    song_name         = sys.argv[1] # Your song name
    pre_trained_model = sys.argv[2] # Pretrained model is "joint-model-all-2500"
    
    print("Now, it is for {}".format(classifier_type))
    
    if classifier_type == "using":
        """
        "using" in here is using the "late-fusion model"
        """
        device        = torch.device('cuda')
        model         = Joint_ShortChunkCNN_Res().to(device) # Loading model
        model         = nn.DataParallel(model)
        print(model)
        input_shape_1 = (-1,384,50)    # Input Autocorrelation Tempogram shape
        input_shape_2 = (-1,193,50)    # Input Fourier Tempogram shape
        input_shape_3 = (-1,1,128,200) # Input Mel-spectrogram shape
        
        # Calculate the audio to npy and loading data into datalaoder
        print("Start to extract features and set up the dataset ...")
        save_feature(data_name = song_name)
        feature_t, feature_f, feature_m, z_test = set_the_dataset_using()
        data_loader = torch.utils.data.DataLoader(
                
                ConcatDataset(feature_t, feature_f, feature_m),                                           
                batch_size = 1,
                shuffle = False
                
                )
        print("Done!")
        print("Start predict the tracks ...")
        
        answers = Using_joint_model(data_loader,
                                    input_shape_1,
                                    input_shape_2,
                                    input_shape_3,
                                    pre_trained_model = pre_trained_model)
        target_test = None
        result  = song_level_test(answers, z_test, target_test, training_use = False)
        print("Done! The result is under the folder and called result.csv")
