from sklearn import svm
import numpy as np
from data import *

class KSVMWrap():
    def __init__(self, X, y, param_svm_gamma='auto', C=1.0):
        self.model = svm.SVC(kernel='rbf', C=C, probability=True, gamma=param_svm_gamma)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

    @property
    def support(self):
        return self.model.support_




if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_
  X, Y_ = sample_gmm_2d(6,2,10)
  # X, Y_ = sample_gauss_2d(3,100)
  Yoh_ = class_to_onehot(Y_)

  # definiraj model:
  ksvm = KSVMWrap(X=X, y=Y_)

  Y = ksvm.predict(X)
  
  acc, rp, confmat = eval_perf_multi(Y_, Y)

  for i, (recall, precision) in enumerate(rp):
    print(f"Class {i} - Recall: {recall:.4f}, Precision: {precision:.4f}")

  print(f'Accuracy: {acc}')
  # iscrtaj rezultate, decizijsku plohu
  rect = (np.min(X, axis=0), np.max(X, axis=0))

  graph_surface(lambda X: ksvm.predict(X), rect, offset=0.5)
  graph_data(X, Y_, Y, special=[ksvm.support])
  plt.show()