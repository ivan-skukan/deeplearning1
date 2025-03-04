import torch, torch.nn.functional as F
import numpy as np
import data


class fcann2:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim,output_dim)
        self.b2 = np.zeros(output_dim)

        self.num_classes = output_dim

    def train(self,X,Y_,param_niter=1e5,param_delta=0.05,param_lambda=1e-3):
        
        Y = data.class_to_onehot(Y_)

        for i in range(param_niter):
            x = np.maximum(0,X @ self.W1 + self.b1)

            scores = x @ self.W2 + self.b2
            exp_scores = np.exp(scores)
            cum_scores = np.sum(exp_scores,axis=1,keepdims=True)

            probs = exp_scores / cum_scores
            # reds = np.argmax(probs,axis=1)

            dscores = probs - Y
            dW2 = x.T @ dscores / X.shape[0] + param_lambda * self.W2
            db2 = np.mean(dscores, axis=0)

            dW1 = X.T @ (((probs-Y) @ self.W2.T) * (x>0)) / X.shape[0] + param_lambda * self.W1
            db1 = np.mean(((probs-Y) @ self.W2.T) * (x>0), axis=0)

            loss = -np.mean(np.sum(Y * np.log(probs + 1e-9),axis=1))

            if i % 100 == 0:
                print(f"Iteration {i}: loss {loss}")

            
            self.W2 -= param_delta*dW2
            self.b2 -= param_delta*db2
            self.W1 -= param_delta*dW1
            self.b1 -= param_delta*db1

    def fcann2_classify(Y_,Y):
        

if __name__ == "__main__":
    np.random.seed(100)

    X,Y_ = data.sample_gmm_2d(6,2,10)
    model = fcann2(input_dim=X.shape[1], hidden_dim = 5, output_dim = np.nunique(Y_))

