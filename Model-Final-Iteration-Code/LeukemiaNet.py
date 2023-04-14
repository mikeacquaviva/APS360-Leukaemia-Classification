# This file contains the final models for the LeukemiaNet project
# The models are based on the ResNet architecture for feature extraction
# and a deep fully connected network for classification

import torch
import numpy as np
import torch.nn as nn
import gc

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Regularization Hyperparameters
LAMBDA1 = 1     # Multi-Class Classification Loss Regularization
LAMBDA2 = 1     # Binary Classification Loss Regularization

## ResNet Architecture
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights ='ResNet18_Weights.DEFAULT')

# Get only the embedding layers of ResNet
modules = list(resnet.children())[:9]
feature_extractor_res = nn.Sequential(*modules)
for f in feature_extractor_res.parameters():
    f.requires_grad = True    # Allow gradient updates for transfer learning fine-tuning

class LeukemiaNet_Features_Resnet(nn.Module):
    """
    Feature extractor for the LeukemiaNet project.
    Based on the ResNet architecture.
    """
    def __init__(self) -> None:
        super(LeukemiaNet_Features_Resnet, self).__init__()
        self.embeddings = feature_extractor_res
        self.name = "LeukemiaNet Feature Extractor"
    def forward(self, x):
        # Compute embeddings:
        return self.embeddings(x).view(x.shape[0], -1)

feature_extractor = LeukemiaNet_Features_Resnet()

class LeukemiaClassifier(nn.Module):
    """
    This class implements a deep fully connected network for classification
    of the Leukemia dataset.
    """
    def __init__(self) -> None:
        super().__init__()
        self.upscale1 = nn.Linear(512, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.upscale2 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.downscale1 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.downscale2 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.downscale3 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.downscale4 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 64)
        self.downscale5 = nn.Linear(64, 16)
        self.fc7 = nn.Linear(16, 16)
        self.downscale6 = nn.Linear(16, 4)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(16)

        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()
        self.drop3 = nn.Dropout()
        self.drop4 = nn.Dropout()
        self.drop5 = nn.Dropout()
        self.drop6 = nn.Dropout()

    
    def forward(self,x):
        x = x.float()
        x = self.upscale1(x)
        x = nn.ReLU()(x)
        x1 = self.fc1(x)
        x = self.drop1(x)
        x = nn.ReLU()(x + x1)
        x = self.bn1(x)

        x = self.upscale2(x)
        x = nn.ReLU()(x)
        x2 = self.fc2(x)
        x = self.drop2(x)
        x = nn.ReLU()(x + x2)
        x = self.bn2(x)

        x = self.downscale1(x)
        x = nn.ReLU()(x)
        x = self.drop3(x)
        x3 = self.fc3(x)
        x = nn.ReLU()(x + x3)
        x = self.bn3(x)

        x = self.downscale2(x)
        x = nn.ReLU()(x)
        x = self.drop4(x)
        x4 = self.fc4(x)
        x = nn.ReLU()(x + x4)
        x = self.bn4(x)

        x = self.downscale3(x)
        x = nn.ReLU()(x)
        x = self.drop5(x)
        x5 = self.fc5(x)
        x = nn.ReLU()(x + x5)
        x = self.bn5(x)

        x = self.downscale4(x)
        x = nn.ReLU()(x)
        x = self.drop6(x)
        x6 = self.fc6(x)
        x = nn.ReLU()(x + x6)
        x = self.bn6(x)

        x = self.downscale5(x)
        x = nn.ReLU()(x)
        x7 = self.fc7(x)
        x = nn.ReLU()(x + x7)
        x = self.bn7(x)

        x = self.downscale6(x)

        return x.float()

def free_cuda(model):
    """
    Free the GPU memory
    """
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

def accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on the given data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct, total = 0,0
    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        emb = feature_extractor(data)
        output = model(emb)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += data.shape[0]
    return correct / total

def fine_tune(classifier, train_dl, val_dl, feature_extractor = feature_extractor, epochs = 50, initial_lr = 5e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    feature_extractor = feature_extractor.to(device)
    optimizer1 = torch.optim.Adam(classifier.parameters(), lr=initial_lr)
    optimizer2 = torch.optim.Adam(feature_extractor.parameters(), lr=initial_lr)
    criterion_classes = nn.CrossEntropyLoss()
    criterion_super = nn.BCELoss()
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, 0.98)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, 0.98)

    train_accuracy, train_loss = np.zeros(epochs), np.zeros(epochs)
    val_accuracy = np.zeros(epochs)

    print("Fine-tuning for %d epochs using %s..."%(epochs,device))

    for epoch in range(epochs):
        running_loss = 0.0
        processed = 0
        for img, label in iter(train_dl):
            classifier.train()
            feature_extractor.train()
            img = img.to(device)
            label = label.to(device)
            emb = feature_extractor(img)
            out = classifier(emb)
            loss = LAMBDA1 * criterion_classes(out, label)

            # Ensure that emb and label have the same batch size
            batch_size = emb.shape[0]

            # Group the classes into 2 groups -- cancerous and benign
            probs = torch.nn.Softmax(dim=1)(out)
            grouped_probs = torch.zeros((batch_size, 2), dtype=float)
            grouped_probs[:, 0] = probs[:, 0]
            grouped_probs[:, 1] = probs[:, 1:].sum(dim=1)

            binary_label = torch.zeros((batch_size, 2), dtype=float)
            for i in range(batch_size):
                if label[i] == 0:
                    binary_label[i,0] = 1
                    binary_label[i,1] = 0
                else:
                    binary_label[i,0] = 0                
                    binary_label[i,1] = 1        

            # Compute the loss
            loss += LAMBDA2 * criterion_super(grouped_probs, binary_label)

            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            running_loss += loss.cpu().item()
            processed += 1
            train_loss[epoch] = running_loss/processed

            with torch.no_grad():
                classifier.eval()
                train_accuracy[epoch] = accuracy(classifier, train_dl)
                val_accuracy[epoch] = accuracy(classifier, val_dl)
            print("Epoch %d |\t Loss: %.2f |\t Training Accuracy: %.2f |\t Validation Accuracy: %.2f"%(epoch+1,train_loss[epoch],train_accuracy[epoch],val_accuracy[epoch]))
            scheduler1.step()
            scheduler2.step()

    torch.save(classifier.state_dict(), "Classifier_tuned.pt")
    torch.save(feature_extractor.state_dict(), "FeatureExtractor_tuned.pt")
    return train_loss, train_accuracy, val_accuracy

def draw_confusion_matrix(predicted_labels, actual_labels):

    cm = confusion_matrix(actual_labels, predicted_labels)
    cm_df = pd.DataFrame(cm,
                     index = ['BENIGN','EARLY','PRE', 'PRO'], 
                     columns = ['BENIGN','EARLY', 'PRE', 'PRO'])

    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix for Leukemia Classifier with ResNet')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

def print_model_report(predicted_labels, actual_labels, class_names):
    print(classification_report(actual_labels, predicted_labels, target_names=class_names))