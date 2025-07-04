import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from zad2 import evaluate


class RNNSentimentClassifier(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, n_layers=2, pad_idx=0, embedding_matrix=None, rnn_type='gru', use_attention=False):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

    if embedding_matrix is not None:
      self.embedding = embedding_matrix
      self.embedding.weight.requires_grad = False  

    if rnn_type == 'gru':
      self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=False, bidirectional=False)
    elif rnn_type == 'lstm':
      self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=False, bidirectional=False)
    elif rnn_type == 'rnn':
      self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=False, bidirectional=False)
    
    self.use_attention = use_attention

    if use_attention:
      self.attention = nn.Linear(hidden_dim, hidden_dim//2)
      self.v = nn.Linear(hidden_dim//2, 1, bias=False)

    self.fc1 = nn.Linear(hidden_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, text, lengths):
    embedded = self.embedding(text)
    embedded = embedded.permute(1, 0, 2)  #  [seq_len, batch_size, emb_dim]

    packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False)
    packed_output, hidden = self.rnn(packed)

    if isinstance(hidden, tuple):
      hidden = hidden[0]

    if self.use_attention:
      out, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
      output = out.permute(1, 0, 2)  

      e = torch.tanh(self.attention(output))
      scores = self.v(e).squeeze(2)  
      a = F.softmax(scores, dim=1)  
      context = torch.bmm(a.unsqueeze(1), output)
      out = context.squeeze(1)  
    else:
      out = hidden[-1]  

    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out.squeeze(1)  

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10):
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for texts, labels, lengths in train_loader:
      optimizer.zero_grad()
      outputs = model(texts, lengths)
      loss = criterion(outputs, labels.float())
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

      preds = torch.round(torch.sigmoid(outputs))
      correct += (preds == labels).sum().item()
      total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print(f'Train: Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    loss, acc, f1, confmat = evaluate(model, val_loader, criterion)

    print(f'Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}')
  print(f'Confusion Matrix:\n{confmat}')

  loss, acc, f1, confmat = evaluate(model, test_loader, criterion)
  print(f'Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}')


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

    model = RNNSentimentClassifier(len(train.vocab),300, 150, embedding_matrix=train.embedding_matrix, rnn_type='lstm')
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=5)

