import torch
from utils import *

if __name__ == '__main__':
  train = NLPDataset('sst/sst_train_raw.csv')
  # valid = NLPDataset('sst/sst_valid_raw.csv')
  # test = NLPDataset('sst/sst_test_raw.csv')

  # print(train.counter)
  # print(train.label_counter)
  train_loader = DataLoader(train, batch_size=20, shuffle=True, collate_fn=pad_collate_fn)

  texts, labels, lengths = next(iter(train_loader))

  print(texts)
  print(labels)
  print(lengths)