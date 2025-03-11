import torch
from torch import nn
import torch.optim as optim
from data import *


class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """

    # inicijalizirati parametre (koristite nn.Parameter):
    # imena mogu biti self.W, self.b
    # ...
    super(PTLogreg, self).__init__()
    self.W = nn.Parameter(torch.randn(D,C), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(C), requires_grad=True)


  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    #   koristiti: torch.mm, torch.softmax
    # ...
    mult = torch.mm(X, self.W) + self.b

    return torch.softmax(mult, dim=1)

  def get_loss(self, X, Yoh_, param_lambda = 0.1):
    # formulacija gubitka
    #   koristiti: torch.log, torch.exp, torch.sum
    #   pripaziti na numerički preljev i podljev
    # ...
    probs = self.forward(X)
    logprobs = torch.log(probs + 1e-8)

    loss = -torch.sum(Yoh_ * logprobs, dim=1).mean()
    loss += (param_lambda / 2) * torch.sum(self.W ** 2)
    # loss = -torch.mean(torch.sum(Yoh_ * logprobs, dim=1))

    return loss


def train(model, X, Yoh_, param_niter, param_delta):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
  """
  
  # inicijalizacija optimizatora
  # ...

  # petlja učenja
  # ispisujte gubitak tijekom učenja
  # ...

  optimizer = optim.SGD(model.parameters(),lr=param_delta)

  for i in range(param_niter):
    #probs = model(X) # redundant?

    loss = model.get_loss(X,Yoh_)

    loss.backward()

    optimizer.step()

    if i % 50 == 0:
        print(f"Iter {i}; Loss {loss.item()}")

    optimizer.zero_grad()


def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  # ulaz je potrebno pretvoriti u torch.Tensor
  # izlaze je potrebno pretvoriti u numpy.array
  # koristite torch.Tensor.detach() i torch.Tensor.numpy()
  with torch.no_grad():
    logits = model(X)

  return np.argmax(logits.detach().numpy(), axis=1)



if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_
  #X,Y_ = sample_gmm_2d(2,2,10)
  X, Y_ = sample_gauss_2d(3,100)
  X = torch.tensor(X, dtype=torch.float32)
  Yoh_ = class_to_onehot(Y_)
  Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)

  # definiraj model:
  ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Yoh_, 1000, 0.2)

  # dohvati vjerojatnosti na skupu za učenje
  Y = eval(ptlr, X)
  # Y = np.argmax(probs, axis=1)
  # print(Y)
  # ispiši performansu (preciznost i odziv po razredima)
  acc, rp, confmat = eval_perf_multi(Y_, Y)

  for i, (recall, precision) in enumerate(rp):
    print(f"Class {i} - Recall: {recall:.4f}, Precision: {precision:.4f}")

  # iscrtaj rezultate, decizijsku plohu
  rect = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))

  graph_surface(lambda X: eval(ptlr, torch.tensor(X, dtype=torch.float32)), rect, offset=0.5)
  graph_data(X.numpy(), Y_, Y, special=[])
  plt.show()
