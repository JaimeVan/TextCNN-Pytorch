'''
Created by JaimeVan
2024-10-09
'''
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report

#%% Training the Model
def train(m, device, train_loader, optimizer, epoch, max_epoch):
    corrects, train_loss = 0.0,0
    for indices, target in train_loader:
        input_tensor = indices.to(device)
        
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
    all_targets = []
    all_predictions = []
    corrects, test_loss = 0.0, 0
    
    for indices, target in test_loader:
        input_tensor = indices.to(device)
        target = target.to(device)

        # Forward pass
        logit = m(input_tensor)
        loss = F.cross_entropy(logit, target)

        # Update total loss
        test_loss += loss.item()

        # Get predicted class and target class (from one-hot to class index)
        result = torch.max(logit, 1)[1]
        target_result = torch.max(target, 1)[1]

        # Collect all targets and predictions for metrics calculation
        all_targets.extend(target_result.cpu().numpy())
        all_predictions.extend(result.cpu().numpy())

        # Calculate correct predictions
        corrects += (result == target_result).sum()

    # Calculate test loss and accuracy
    size = len(test_loader.dataset)
    test_loss /= size 
    accuracy = 100.0 * corrects / size

    # Generate classification report using sklearn
    report = classification_report(all_targets, all_predictions, output_dict=True)

    # Convert classification report to DataFrame and save to CSV
    df = pd.DataFrame(report).transpose()
    df.to_csv('./classification_report.csv', index=True)

    return test_loss, accuracy