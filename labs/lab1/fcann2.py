import numpy as np
import data


def fcann2_train(X,Y_,hidden, param_niter,param_delta,param_lambda):
    num_classes = len(np.unique(Y_))

    Y = data.class_to_onehot(Y_)

    W1 = np.random.randn(X.shape[1], hidden)
    b1 = np.zeros((1,hidden)) # or just hidden

    W2 = np.random.randn(hidden, num_classes)
    b2 = np.zeros((1,num_classes))

    for i in range(param_niter):
        scores_1 = np.dot(X,W1) + b1
        scores_1_relu = np.maximum(0,scores_1)
        scores_2 = np.dot(scores_1_relu,W2) + b2

        probs = np.exp(scores_2) / np.sum(np.exp(scores_2), axis=1, keepdims=True)
        logprobs = -np.log(probs + 1e-8)

        loss = -np.mean(Y*logprobs)

        if i % 100 == 0:
            print(f"Iter {i}; loss {loss:.4f}")

        grad_scores_2 = probs - Y

        grad_W2 = np.dot(scores_1_relu.T, grad_scores_2) / X.shape[0] + param_lambda * W2
        grad_b2 = np.sum(grad_scores_2, axis=0, keepdims=True) / X.shape[0]

        grad_scores_1_relu = np.dot(grad_scores_2, W2.T)
        grad_scores_1 = grad_scores_1_relu * (scores_1 > 0) 

        grad_W1 = np.dot(X.T, grad_scores_1) / X.shape[0] + param_lambda * W1
        grad_b1 = np.sum(grad_scores_1, axis=0, keepdims=True) / X.shape[0]

        W2 -= param_delta * grad_W2
        b2 -= param_delta * grad_b2

        W1 -= param_delta * grad_W1
        b1 -= param_delta * grad_b1


    return W1,b1,W2,b2



def fcann2_classify(X,W1,b1,W2,b2):
    scores_1 = np.dot(X,W1) + b1
    relu = np.maximum(0, scores_1)
    scores_2 = np.dot(relu, W2) + b2
    probs = np.exp(scores_2) / np.sum(np.exp(scores_2), axis=1, keepdims=True)

    return probs

def fcann2_decfun(W1,b1,W2,b2):
    def classify(X):
        return np.argmax(fcann2_classify(X,W1,b1,W2,b2), axis=1)
    return classify


if __name__ == '__main__':
    np.random.seed(100)
    K = 6
    C = 2
    N = 10
    X,Y_ = data.sample_gmm_2d(K,C,N)

    param_niter = int(1e5)
    param_delta = 0.05
    param_lambda = 1e-3
    hidden_layer = 5

    W1,b1,W2,b2 = fcann2_train(X,Y_,hidden_layer, param_niter, param_delta, param_lambda)
    Y = np.argmax(fcann2_classify(X,W1,b1,W2,b2), axis=1)

    func = fcann2_decfun(W1,b1,W2,b2)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(func,bbox,offset=0.5)
    data.graph_data(X,Y_,Y)
    data.plt.show()