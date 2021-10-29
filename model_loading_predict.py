#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:43:15 2021

@author: maclab
"""

import torch
from tqdm import tqdm
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import confusion_matrix

def Using_joint_model(testing_loader, 
                      input_shape_1, 
                      input_shape_2, 
                      input_shape_3,
                      pre_trained_model = "joint-model-all-2500.pkl"):
    model = torch.load("./model_pkl/{}".format(pre_trained_model))
    answers = []
    model.eval()
    with torch.no_grad():
        for (matrix1, matrix2, matrix3) in tqdm(testing_loader):
            matrix1   = matrix1[0].cuda()
            matrix2   = matrix2[0].cuda()
            matrix3   = matrix3[0].cuda()
            matrix1   = Variable(matrix1.view(input_shape_1)).cuda()
            matrix2   = Variable(matrix2.view(input_shape_2)).cuda()
            matrix3   = Variable(matrix3.view(input_shape_3)).cuda()
            outputs   = model(matrix1, matrix2, matrix3).cuda()
            predicted = torch.max(outputs.data, 1)[1]
            answers.append(int(predicted[0]))
    print("Done!!!")      
    return answers

def song_level_test(answers, z_test, targets_test_1, training_use = True):
    
    if not training_use:
        
        songidtest = pd.Categorical(z_test)
        songidtest = songidtest.categories
        genre_list = pd.read_csv("genre_list.csv")
        genre_list = list(genre_list["0"])
        test_      = pd.DataFrame({"answer": answers, "id" : z_test})
        predict    = []
        song       = []
        for i in tqdm(range(len(songidtest))):
            temp_number = test_[test_["id"] == songidtest[i]]["answer"].value_counts().idxmax()
            predict.append(genre_list[temp_number])
            song.append(songidtest[i])
        result = pd.DataFrame({"predict" : predict, "song" : song})
        result.to_csv("./result.csv", sep=",")
        
        return result
    
    else :
        songidtest = pd.Categorical(z_test)
        songidtest = songidtest.categories
        test_      = pd.DataFrame({"answer": answers, "targets" : targets_test_1, "id" : z_test})
        id_already = ["123"]
        counts     = 0
        target     = []
        predict    = []
        song       = []
        for i in tqdm(range(len(answers))):
            if not test_["id"][i] in id_already:
                temp_ans = test_[test_['id'] == test_["id"][i]].value_counts().idxmax()
                if temp_ans[0] == temp_ans[1]:
                    counts += 1
                else :
                    counts += 0
            else :
                continue
            id_already.append(test_['id'][i])
        print("done! song_level acc : {}".format(counts/len(test_["id"].value_counts())))
        for i in tqdm(range(len(songidtest))):
            predict.append(test_[test_["id"] == songidtest[i]]["answer"].value_counts().idxmax())
            target.append(test_[test_["id"]  == songidtest[i]]["targets"].value_counts().idxmax())
            song.append(songidtest[i])
        ConfusionMatrix = confusion_matrix(target, predict) 
        result = pd.DataFrame({"target" : target, "predict" : predict, "song" : song})
        result.to_csv("song_level_SCcnn256_result.csv", sep=",")
        
        return counts/len(test_["id"].value_counts()), ConfusionMatrix, result

