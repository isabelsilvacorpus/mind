import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.optim as optim

class NewsDataset(Dataset):
    '''
    vecs: embedding for each news item 
    labels: cateogry label for each news item 
    news id: ID of each news item 
    '''
    def __init__(self, vecs, labels):
        self.vecs = torch.tensor(vecs, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.int64)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        vecs = self.vecs[idx]
        label = self.labels[idx]
        return vecs, label

class multiclass_classifier(nn.Module):
    '''
    Simple sequential network for classification (17 classes)
    input: 100-d vector 
    output: likelihood of each class 
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 17)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.output(x)
    
class multiclass_classifier_title(nn.Module):
    '''
    Simple sequential network for classification (17 classes) (intended for sbert title embeddings)
    input: 384-d SBERT sentence (title) embeddings
    output: likelihood of each class 
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 17)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.output(x)
    
class multiclass_classifier_concat(nn.Module):
    '''
    Simple sequential network for classification (17 classes) for concat embeddings
    input: 484-d vector
    output: likelihood of each class 
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(484, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 17)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.output(x)

def train_model(model, loss_fn, optimizer, n_epochs, train_loader): 
    '''
    model: nn.Module 
    train_loader: DataLoader 
    '''
    model.train()
    for epoch in range(n_epochs):
        train_loss = 0.0
        correct = 0.0
        counter = 0.0

        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(y_pred.data, 1)
            correct += (preds == y_batch).float().sum()
            counter += y_batch.size(0)
        
        train_acc = correct/counter
        print(f'epoch: {epoch}, training loss: {train_loss}, training accuracy: {train_acc}')

    return model, train_loss, train_acc 

def validate_model(model, loss_fn, test_loader):
    '''
    model: trained nn.Module 
    test_loader: DataLoader
    '''
    model.eval()
    total_loss = 0.0 
    correct = 0.0 
    counter = 0.0

    for X_test, y_test in test_loader:
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        total_loss += loss.item()

        _, preds = torch.max(y_pred.data, 1)
        counter += y_test.size(0)
        correct += (preds == y_test).float().sum()
    test_acc = correct/counter
    return test_acc,total_loss