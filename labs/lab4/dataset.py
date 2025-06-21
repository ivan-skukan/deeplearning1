from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import torchvision.transforms as transforms


class MNISTMetricDataset(Dataset):
  def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
    super().__init__()
    assert split in ['train', 'test', 'traineval']
    self.root = root
    self.split = split

    transform = transforms.ToTensor()  
    mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True, transform=transform)
    self.images, self.targets = mnist_ds.data, mnist_ds.targets  
    self.images = self.images.unsqueeze(1).float() / 255. 
    self.classes = list(range(10))

    if remove_class is not None:
      assert remove_class in self.classes, f"Class {remove_class} not in MNIST classes."
      self.images = self.images[self.targets != remove_class]
      self.targets = self.targets[self.targets != remove_class]
      self.classes.remove(remove_class)

    self.target2indices = defaultdict(list)
    for i in range(len(self.targets)):
      self.target2indices[self.targets[i].item()].append(i)

  def _sample_negative(self, index):
    target = self.targets[index].item()
    negative_class = choice([c for c in self.classes if c != target])
    return choice(self.target2indices[negative_class])

  def _sample_positive(self, index):
    target = self.targets[index].item()
    positive_indices = [i for i in self.target2indices[target] if i != index]
    return choice(positive_indices)

  def __getitem__(self, index):
    anchor = self.images[index]
    target_id = self.targets[index].item()

    if self.split in ['traineval', 'val', 'test']:
      return anchor, target_id
    else:
      positive = self.images[self._sample_positive(index)]
      negative = self.images[self._sample_negative(index)]
      return anchor, positive, negative, target_id

  def __len__(self):
    return len(self.images)
