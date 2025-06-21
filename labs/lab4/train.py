import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding
from identitymodel import IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


def identity(train_loader, test_loader, traineval_loader, model, device, epochs, num_classes, emb_size):
    print("Identity. No training.")
    model.eval()
    representations = compute_representations(model, train_loader, num_classes, emb_size, device)
    if EVAL_ON_TRAIN:
        print("Evaluating on training set...")
        acc1 = evaluate(model, representations, traineval_loader, device)
        print(f"Train Top1 Acc: {round(acc1 * 100, 2)}%")
    if EVAL_ON_TEST:
        print("Evaluating on test set...")
        acc1 = evaluate(model, representations, test_loader, device)
        print(f"Test Accuracy: {acc1 * 100:.2f}%")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "/tmp/mnist"  
    ds_train = MNISTMetricDataset(mnist_download_root, split='train', remove_class=0)
    ds_train_for_representation = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    train_loader_for_representation = DataLoader(
        ds_train_for_representation,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    #model = IdentityModel().to(device)
    model = SimpleMetricEmbedding(1, 32).to(device)
    if isinstance(model, SimpleMetricEmbedding):
      optimizer = torch.optim.Adam(
          model.parameters(),
          lr=1e-3
      )
      emb_size = 32
    else:
      emb_size = 784

    epochs = 3

    if isinstance(model, IdentityModel):
      identity(train_loader, test_loader, traineval_loader, model, device, epochs, num_classes, emb_size)
    else:
      for epoch in range(epochs):
          print(f"Epoch: {epoch}")
          t0 = time.time_ns()
          train_loss = train(model, optimizer, train_loader, device)
          print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
          if EVAL_ON_TEST or EVAL_ON_TRAIN:
              print("Computing mean representations for evaluation...")
              representations = compute_representations(model, train_loader_for_representation, num_classes, emb_size, device)
          if EVAL_ON_TRAIN:
              print("Evaluating on training set...")
              acc1 = evaluate(model, representations, traineval_loader, device)
              print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
          if EVAL_ON_TEST:
              print("Evaluating on test set...")
              acc1 = evaluate(model, representations, test_loader, device)
              print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
          t1 = time.time_ns()
          print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")

      torch.save(model.state_dict(), "/home/ivan/fer/deeplearning1/labs/lab4/model_params_no0.pth")

