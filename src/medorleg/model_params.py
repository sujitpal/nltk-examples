from __future__ import division

import cPickle as pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def train(prefix):
  X = np.exp(np.loadtxt(prefix + "_X.txt"))
  y = np.exp(np.loadtxt(prefix + "_y.txt"))
  model = LinearRegression()
  model.fit(X, y)
  ypred = model.predict(X)
  print prefix, model.coef_, model.intercept_, r2_score(y, ypred)
  pickle.dump(model, open(prefix + ".pkl", 'wb'))
  
def main():
  train("data/regdata/medical")
  train("data/regdata/legal")
  
if __name__ == "__main__":
  main()
