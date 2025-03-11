import torch
import torchvision
import matplotlib.pyplot as plt
import pt_deep
from data import *

dataset_root = '/tmp/mnist'  # change this to your preference

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False, transform=transform)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False, transform=transform)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)


if __name__ == '__main__':
    no_depth = [784,10] # flatten!!!!
    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)
    ptd = pt_deep.PTDeep(no_depth)
    Yoh_train = torch.tensor(class_to_onehot(y_train), dtype=torch.float32)
    Yoh_test  = torch.tensor(class_to_onehot(y_test), dtype=torch.float32)
    pt_deep.train(model=ptd, X=x_train, Yoh_=Yoh_train, param_niter=int(2000), param_delta=0.1)

    Y = pt_deep.eval(model=ptd, X=x_test)

    acc, rp, confmat = eval_perf_multi(y_test.numpy(), Y)

    for i, (recall, precision) in enumerate(rp):
        print(f"Class {i} - Recall: {recall:.4f}, Precision: {precision:.4f}")

    # rect = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))

    # graph_surface(lambda X: eval(ptlr, torch.tensor(X, dtype=torch.float32)), rect, offset=0.5)
    # graph_data(X.numpy(), Y_, Y, special=[])
    # plt.show()