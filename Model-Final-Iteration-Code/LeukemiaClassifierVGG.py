import torch
import numpy as np
import torch.nn as nn
import gc

class LeukemiaClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upscale1 = nn.Linear(25088, 1024)
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

def free_cuda(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

def accuracy(model, data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct, total = 0,0
    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += data.shape[0]
    return correct / total

def train(model, train_dl, val_dl, epochs = 50, initial_lr = 5e-4, bs = 25):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    train_accuracy, train_loss = np.zeros(epochs), np.zeros(epochs)
    val_accuracy = np.zeros(epochs)

    print("Training for %d epochs using %s..."%(epochs,device))

    for epoch in range(epochs):
        running_loss = 0.0
        processed = 0
        for emb, label in iter(train_dl):
            model.train()
            emb = emb.to(device)
            label = label.to(device)
            out = model(emb)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.cpu().item()
            processed += 1
            train_loss[epoch] = running_loss/processed

            with torch.no_grad():
                model.eval()
                train_accuracy[epoch] = accuracy(model, train_dl)
                val_accuracy[epoch] = accuracy(model, val_dl)
            print("Epoch %d |\t Loss: %.2f |\t Training Accuracy: %.2f |\t Validation Accuracy: %.2f"%(epoch+1,train_loss[epoch],train_accuracy[epoch],val_accuracy[epoch]))
            scheduler.step()

    torch.save(model.state_dict(), "Trained_Model_Iteration_1")
    return train_loss, train_accuracy, val_accuracy
