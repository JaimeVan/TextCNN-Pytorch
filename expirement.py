'''
Created by JaimeVan
2024-10-09
'''
import torch
import sys
import numpy as np
import torch.nn.functional as F

#%% Training the Model
def train(m, device, train_loader, optimizer, epoch, max_epoch):
    corrects, train_loss = 0.0,0
    for indices, target in train_loader:

        input_tensor = torch.tensor(indices, dtype=torch.long).to(device)
        
        target = target.to(device)

        optimizer.zero_grad()
        logit = m(input_tensor)

        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.item()

        result = torch.max(logit,1)[1]
        target_result = torch.max(target, 1)[1]

        corrects += (result == target_result).sum()

    
    size = len(train_loader.dataset)
    train_loss /= size 
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy
    
def valid(m, device, test_loader):
    corrects, test_loss = 0.0,0
    for indices, target in test_loader:
        input_tensor = torch.tensor(indices, dtype=torch.long).to(device)

        target = target.to(device)
        
        logit = m(input_tensor)
        loss = F.cross_entropy(logit, target)

        
        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        target_result = torch.max(target, 1)[1]
        corrects += (result == target_result).sum()
    
    size = len(test_loader.dataset)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy