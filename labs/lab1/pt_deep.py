import torch
from torch import nn
import torch.optim as optim
from data import *


class PTDeep(nn.Module):
    def __init__(self, layers, activation=torch.relu):
        super(PTDeep, self).__init__()

        self.weights = nn.ParameterList([nn.Parameter(torch.randn(layers[i-1],layers[i])) for i in range(1,len(layers))])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(layers[i])) for i in range(1,len(layers))])
        self.activation = activation

    def forward(self, X):
        x = X
        for i in range(len(self.weights)-1):
            x = torch.mm(x, self.weights[i]) + self.biases[i]
            x = self.activation(x)  
        x = torch.mm(x, self.weights[-1]) + self.biases[-1]
        return x

    def get_loss(self, X, Yoh_, param_lambda = 1e-4):
        
        vals = self.forward(X)
        probs = torch.softmax(vals, dim=1)
        logprobs = torch.log(probs + 1e-8)

        loss = -torch.sum(Yoh_ * logprobs, dim=1).mean()
        reg = sum(torch.sum(W**2) for W in self.weights)
        loss += (param_lambda / 2) * reg

        return loss

def train(model, X, Yoh_, param_niter, param_delta,param_lambda=1e-4):
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
    #probs = model(X) 

    loss = model.get_loss(X,Yoh_,param_lambda=param_lambda)

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


def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_
  X, Y_ = sample_gmm_2d(6,2,10)
  # X, Y_ = sample_gauss_2d(3,100)
  X = torch.tensor(X, dtype=torch.float32)
  Yoh_ = class_to_onehot(Y_)
  Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)

  # definiraj model:
  layers = [2,10,10,2]
  ptlr = PTDeep(layers, torch.relu)
  print('Model params:', count_params(ptlr))
  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Yoh_, 10000, 0.1)

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
