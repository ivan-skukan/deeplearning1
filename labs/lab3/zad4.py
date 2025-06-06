import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from zad2 import evaluate
from zad3 import RNNSentimentClassifier
from utils import NLPDataset, pad_collate_fn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_single_config(config, train_loader, val_loader, test_loader, device, seed=42):
    set_seed(seed)

    model = RNNSentimentClassifier(
        vocab_size=len(train_loader.dataset.vocab),
        embedding_dim=300,
        hidden_dim=config['hidden_size'],
        output_dim=1,
        n_layers=config['num_layers'],
        pad_idx=0,
        embedding_matrix=train_loader.dataset.embedding_matrix,
        rnn_type=config['rnn_type']
    ).to(device)

    if config['num_layers'] > 1:
        model.rnn.dropout = config['dropout']

    model.rnn.bidirectional = config['bidirectional']

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(3):  # short for fast testing
        model.train()
        for texts, labels, lengths in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

    val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion)
    return val_loss, val_acc, val_f1


def run_all_configs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = NLPDataset('sst/sst_train_raw.csv')
    valid = NLPDataset('sst/sst_valid_raw.csv')
    test = NLPDataset('sst/sst_test_raw.csv')

    valid.setVocab(train.vocab)
    valid.embedding_matrix = train.embedding_matrix
    test.setVocab(train.vocab)
    test.embedding_matrix = train.embedding_matrix

    train_loader = DataLoader(train, batch_size=10, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(valid, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    base_config = {
        'rnn_type': 'gru',
        'hidden_size': 150,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': False
    }

    results = []

    print("Running grid search for RNN configurations...")
    for rnn_type in ['rnn', 'gru', 'lstm']:
        cfg = base_config.copy()
        cfg['rnn_type'] = rnn_type
        loss, acc, f1 = run_single_config(cfg, train_loader, val_loader, test_loader, device)
        results.append({**cfg, 'val_loss': loss, 'val_acc': acc, 'val_f1': f1})

    print("Testing different hidden sizes...")
    for hs in [64, 150, 300]:
        cfg = base_config.copy()
        cfg['hidden_size'] = hs
        loss, acc, f1 = run_single_config(cfg, train_loader, val_loader, test_loader, device)
        results.append({**cfg, 'val_loss': loss, 'val_acc': acc, 'val_f1': f1})

    print("Testing different number of layers...")
    for nl in [1, 2, 3]:
        cfg = base_config.copy()
        cfg['num_layers'] = nl
        loss, acc, f1 = run_single_config(cfg, train_loader, val_loader, test_loader, device)
        results.append({**cfg, 'val_loss': loss, 'val_acc': acc, 'val_f1': f1})

    print("Testing different dropout rates...")
    for dr in [0.0, 0.3, 0.5]:
        cfg = base_config.copy()
        cfg['dropout'] = dr
        cfg['num_layers'] = 2
        loss, acc, f1 = run_single_config(cfg, train_loader, val_loader, test_loader, device)
        results.append({**cfg, 'val_loss': loss, 'val_acc': acc, 'val_f1': f1})

    print("Testing bidirectional RNNs...")
    for bi in [False, True]:
        cfg = base_config.copy()
        cfg['bidirectional'] = bi
        loss, acc, f1 = run_single_config(cfg, train_loader, val_loader, test_loader, device)
        results.append({**cfg, 'val_loss': loss, 'val_acc': acc, 'val_f1': f1})

    df = pd.DataFrame(results)
    df.to_csv('rnn_grid_results.csv', index=False)
    print("Saved results to rnn_grid_results.csv")


if __name__ == '__main__':
    run_all_configs()
