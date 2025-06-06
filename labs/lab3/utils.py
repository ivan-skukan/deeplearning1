import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import Counter


@dataclass
class Instance:
  tokens: list[str]
  label: str

class NLPDataset(Dataset):
  def __init__(self, path: str):
    self.instances = []
    with open(path, encoding='utf-8') as f:
      reader = pd.read_csv(f, header=None).values.tolist()
      for row in reader:
        text, label = row[0], row[1]
        tokens = text.lower().split() 
        self.instances.append(Instance(tokens=tokens, label=label))

    self.counter = self.build_frequency_dict()
    self.label_counter = self.build_frequency_dict(field="label")
    self.vocab = Vocab(self.counter)
    self.vocab_labels = Vocab(self.label_counter, special_tokens=None)

    self.embedding_matrix = get_embedding_matrix(self.vocab, './glove/sst_glove_6b_300d.txt', embedding_dim=300)

  def setVocab(self, vocab):
    self.vocab = vocab

  def __len__(self):
    return len(self.instances)

  def __getitem__(self, idx):
    instance = self.instances[idx]
    numericalized = self.vocab.encode(instance.tokens)
    labelidx = self.vocab_labels.encode(instance.label)
    return numericalized, labelidx

  def build_frequency_dict(self, field="tokens"):
    counter = Counter()
    for instance in self.instances:
      items = getattr(instance, field)
      if isinstance(items, str):
        items = [items]  # Wrap string label into list for Counter.update
      counter.update(items)
    return counter

class Vocab:
  def __init__(self, frequencies, max_size=-1, min_freq=1, special_tokens = ['<PAD>', '<UNK>']):
      
    if special_tokens:
      self.stoi = {token: idx for idx, token in enumerate(special_tokens)}
    else:
      self.stoi = {}
    self.itos = []

    sorted_items = sorted([item for item, freq in frequencies.items() if freq >= min_freq], key=lambda x: (-frequencies[x], x))

    if max_size > 0:
        sorted_items = sorted_items[:max_size - 2]
    for item in sorted_items:
      self.stoi[item] = len(self.stoi)
      self.itos.append(item)

  def encode(self, tokens):
    if isinstance(tokens, str):
      return torch.tensor(self.stoi.get(tokens, self.stoi.get('<UNK>', -1)))
    elif isinstance(tokens, list):
      return torch.tensor([self.stoi.get(token, self.stoi.get('<UNK>', -1)) for token in tokens])
  
  def __len__(self):
      return len(self.stoi)


def get_embedding_matrix(vocab: Vocab, glove_path=None, embedding_dim=300):
  vocab_size = len(vocab)
  embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim)).astype(np.float32)
  embedding_matrix[0] = np.zeros(embedding_dim)  # <PAD> token at index 0

  if glove_path:
    print(f"Loading GloVe vectors from {glove_path}...")
    glove = {}

    with open(glove_path, encoding="utf-8") as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) != embedding_dim + 1:
          continue  # skip malformed lines
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        glove[word] = vector

    found = 0
    for word, idx in vocab.stoi.items():
      if word in glove:
        embedding_matrix[idx] = glove[word]
        found += 1

    print(f"Found {found}/{vocab_size} words in GloVe.")

  embedding_tensor = torch.tensor(embedding_matrix)
  return nn.Embedding.from_pretrained(embedding_tensor, padding_idx=0, freeze=bool(glove_path))


def pad_collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, lengths
