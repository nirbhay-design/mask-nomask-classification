import seaborn as sns
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import roc_curve,auc, precision_score,precision_recall_curve,recall_score,precision_recall_fscore_support,confusion_matrix
import numpy as np
from prettytable import PrettyTable
#print(torch.cuda.is_available())
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
#print(torch.cuda.get_device_properties(0).total_memory)
#print(torch.cuda.memory_allocated())
gpu_id = 2

class Predict_Model_class(object):

    def __init__(self,py_model):
        self.model = py_model    

        self.mapping = {0:'masked',1:'unmasked'}

    def Normalize(self,data):
#         data = data/255
        for i in range(data.shape[0]):
            mean = torch.mean(data[i],dim = [1,2])
            std = torch.std(data[i],dim=[1,2])
            transform = transforms.Compose([transforms.Normalize(mean,std)])
            data[i] = transform(data[i])
        return data

    def evaluate_img(self,image):
        self.model.eval()
        image = self.Normalize(image)
        scores = self.model(image)
        scores = F.softmax(scores,dim=1)
        _,predicted = torch.max(scores,dim=1)

        
        return scores,self.mapping[int(predicted.item())]


    def evaluate_batch(self,batch,labels):
        
        self.model.eval()

        correct = 0;samples=0;

        with torch.no_grad():
            scores = self.model(batch)

            scores =F.softmax(scores,dim=1)
            _,predicted = torch.max(scores,dim = 1)
            correct += (predicted == labels).sum()
            samples += scores.shape[0]

            # torch.cuda.empty_cache(self.gpu_id)
            self.model.train()

        return correct/samples

    def print_params(self):
        # table = PrettyTable(["layer","parameters"])

        total_parameters = 0
        for name,parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            # table.add_row([name,param])
            total_parameters += param

        # print(table)
        print(f"total_trainable_parameters are : {total_parameters}")

    
        

    

   

   


















































