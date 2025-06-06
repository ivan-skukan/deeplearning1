import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import *


class LameModel(nn.Module):
  def __init__(self, embedding_layer: nn.Embedding):
    super().__init__()
    self.embedding = embedding_layer
    self.fc1 = nn.Linear(300, 150)
    self.fc2 = nn.Linear(150, 150)
    self.fc3 = nn.Linear(150, 1)  

  def forward(self, x, lengths):
    embedded = self.embedding(x)  
    mean_pooled = embedded.mean(dim=1)  
    x = F.relu(self.fc1(mean_pooled))  
    x = F.relu(self.fc2(x))            
    x = self.fc3(x)                    
    return x.squeeze(1)  


def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for texts, labels, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * texts.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate(model, data_loader, loss_fn):
  model.eval()
  total_loss = 0.0
  all_labels = []
  all_preds = []

  with torch.no_grad():
    for texts, labels, lengths in data_loader:
      outputs = model(texts, lengths)
      loss = loss_fn(outputs, labels.float())
      total_loss += loss.item() * texts.size(0)

      preds = torch.sigmoid(outputs) > 0.5
      all_labels.extend(labels.cpu().numpy())
      all_preds.extend(preds.cpu().numpy())
  total_loss /= len(data_loader.dataset)
  accuracy = accuracy_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds, average='binary')
  confmat = confusion_matrix(all_labels, all_preds)
  return total_loss, accuracy, f1, confmat

def train_model(model, train_loader, val_loader, test_loader, optimizer, num_epochs=5):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
      trainloss = train_epoch(model, train_loader, optimizer, loss_fn)
      val_loss, val_acc, val_f1, val_confmat = evaluate(model, val_loader, loss_fn)

      print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {trainloss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
      # print(f'Confusion Matrix:\n{val_confmat}')

    test_loss, test_acc, test_f1, test_confmat = evaluate(model, test_loader, loss_fn)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    
if __name__ == '__main__':
    train = NLPDataset('sst/sst_train_raw.csv')
    valid = NLPDataset('sst/sst_valid_raw.csv')
    test = NLPDataset('sst/sst_test_raw.csv')

    valid.setVocab(train.vocab)
    valid.vocab_labels = train.vocab_labels
    valid.embedding_matrix = train.embedding_matrix

    test.setVocab(train.vocab)
    test.vocab_labels = train.vocab_labels
    test.embedding_matrix = train.embedding_matrix

    train_loader = DataLoader(train, batch_size=10, shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    model = LameModel(train.embedding_matrix)
    optimizer = torch.optim.Adam(model.parameters())

    train_model(model, train_loader, valid_loader, test_loader, optimizer, num_epochs=5)