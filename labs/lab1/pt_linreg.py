import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.mean(diff**2)

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    grad_a_manual = (-2 / X.shape[0]) * torch.sum(X * (Y - Y_))
    grad_b_manual = (-2 / X.shape[0]) * torch.sum(Y - Y_)

    assert grad_a_manual == a.grad
    assert grad_b_manual == b.grad

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
    print(f'a grad: {a.grad}; b grad {b.grad}')
    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()
