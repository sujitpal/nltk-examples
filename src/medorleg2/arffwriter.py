import os.path
import numpy as np
import operator

def qq(s):
  return "'" + s + "'"

def save_arff(X, y, vocab, fname):
  aout = open(fname, 'wb')
  # header
  aout.write("@relation %s\n\n" %
    (os.path.basename(fname).split(".")[0]))
  # input variables
  for term in vocab:
    aout.write("@attribute \"%s\" numeric\n" % (term))
  # target variable
  aout.write("@attribute target_var {%s}\n" %
    (",".join([qq(str(int(e))) for e in list(np.unique(y))])))
  # data
  aout.write("\n@data\n")
  for row in range(0, X.shape[0]):
    rdata = X.getrow(row)
    idps = sorted(zip(rdata.indices, rdata.data), key=operator.itemgetter(0))
    if len(idps) > 0:
      aout.write("{%s,%d '%d'}\n" % (
        ",".join([" ".join([str(idx), str(dat)]) for (idx,dat) in idps]),
        X.shape[1], int(y[row])))
  aout.close()
