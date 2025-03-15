import torch
import torchvision
import matplotlib.pyplot as plt
import pt_deep
from data import *
from sklearn import svm

dataset_root = '/tmp/mnist'  # change this to your preference

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True, transform=transform)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
#print(torch.nn.functional.one_hot(y_train, num_classes=-1))
# y_train = torch.nn.functional.one_hot(y_train, num_classes=-1)
# y_test = torch.nn.functional.one_hot(y_test, num_classes=-1)

N = x_train.shape[0]
data_idx = torch.randperm(N)
val_size = N // 5
x_val, y_val = x_train[data_idx[:val_size]], y_train[data_idx[:val_size]]

def train_mb(model, X_train, X_val, Y_train, Y_val, param_batchsize, param_nepochs, param_delta=1e-4,param_lambda=1e-4, track_losses=True):
    if track_losses:
        losses = []
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    variable_step = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4) 

    best_model = None
    min_loss = float('inf')

    y_train_oh = torch.nn.functional.one_hot(Y_train, num_classes=10)

    param_niter = X_train.shape[0] // param_batchsize

    for epoch in range(param_nepochs):
        random_indices = torch.randperm(X_train.shape[0])
        shuffled_X = X_train[random_indices]
        shuffled_Y = y_train_oh[random_indices]

        epoch_loss = 0

        
        for iter in range(0,10000,param_batchsize):

            batch_X = shuffled_X[iter:iter+param_batchsize]
            batch_Y = shuffled_Y[iter:iter+param_batchsize]
            
            optimizer.zero_grad()

            loss = model.get_loss(batch_X, batch_Y)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():

            loss = model.get_loss(X_val, torch.nn.functional.one_hot(Y_val, num_classes=10))
            y = pt_deep.eval(model, X_val)
            # y = np.argmax(y, axis=1)
            acc, pr, M = eval_perf_multi(y, Y_val.numpy())

            if loss < min_loss:
                min_loss = loss.item()
                best_model = model.state_dict()
            


        if track_losses:
            losses.append(epoch_loss)

        print(f'Epoch: {epoch}, Loss: {loss}, Acc: {acc}')
        variable_step.step()

    if track_losses:
        return model, losses
    return model


def plot_weights(model):
    fig,axs = plt.subplots(2,5)
    weights = model.weights[-1].detach().numpy()
    for i in range(2):
        for j in range(5):
            weight_img = weights[:, i*5+j].reshape(28,28)
            # weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min())
            axs[i,j].imshow(weight_img, cmap=plt.get_cmap('gray'))
            axs[i,j].axis('off')
            axs[i,j].set_title(f'{i*5+j}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    models = [[784,10],[784,100,10]]

    batch_size = 64
    niter = 10000
    nepochs = 50


    for model in models:
        print('===================================================================')
        print(f'Model layers: {model}')
        # flatten zbog 784 umjesto 28x28
        x_train = x_train.view(x_train.size(0), -1)
        x_test = x_test.view(x_test.size(0), -1)
        x_val = x_val.view(x_val.size(0), -1)

        ptd = pt_deep.PTDeep(model)

        # Yoh_train = torch.tensor(class_to_onehot(y_train), dtype=torch.float32) 
        # Yoh_test  = torch.tensor(class_to_onehot(y_test), dtype=torch.float32) 
        
        best_model, losses = train_mb(model=ptd, X_train=x_train, X_val=x_val, Y_train=y_train, Y_val=y_val, param_batchsize=batch_size, param_nepochs=nepochs)

        plt.plot(losses)
        plt.title(f'Losses for layers {model}')
        plt.show()

        if len(model) == 2: # 784x10
            plot_weights(ptd)
    
    print('===================================================================')
    print(f'Random model')
    random_model = pt_deep.PTDeep([784,100,10])
    y = pt_deep.eval(random_model,x_test)
    # print(y.shape)
    # print(y_test.numpy().shape)
    acc, pr, mat = eval_perf_multi(y_test.numpy(), y)
    print(f'Accuracy: {acc}')
    print('===================================================================')
    print('Training SVM. This could take a while')
    svm = svm.SVC()
    svm.fit(x_train.numpy(), y_train.numpy())
    y = svm.predict(x_test)
    acc, pr, mat = eval_perf_multi(y_test.numpy(), y) # numpy?
    precision, recall = zip(*pr)
    avg_p = np.mean(precision)
    avg_r = np.mean(recall)

    print(f'Precision: {avg_p}, recall: {avg_r}')
