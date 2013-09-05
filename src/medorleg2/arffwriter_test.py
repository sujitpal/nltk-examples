import sys
import operator

from arffwriter import save_arff
import datetime
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

def load_xy(xfile, yfile):
  pipeline = Pipeline([
    ("count", CountVectorizer(stop_words='english', min_df=0.0,
              binary=False)),
    ("tfidf", TfidfTransformer(norm="l2"))
  ])
  xin = open(xfile, 'rb')
  X = pipeline.fit_transform(xin)
  xin.close()
  yin = open(yfile, 'rb')
  y = np.loadtxt(yin)
  yin.close()
  vocab_map = pipeline.steps[0][1].vocabulary_
  vocab = [x[0] for x in sorted([(x, vocab_map[x]) 
                for x in vocab_map], 
                key=operator.itemgetter(1))]
  return X, y, vocab

def print_timestamp(message):
  print message, datetime.datetime.now()

def main():
  if len(sys.argv) != 5:
    print "Usage: arffwriter_test Xfile yfile trainARFF testARFF"
    sys.exit(-1)
  print_timestamp("started:")
  X, y, vocab = load_xy(sys.argv[1], sys.argv[2])
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
    test_size=0.1, random_state=42)
  save_arff(Xtrain, ytrain, vocab, sys.argv[3])
  save_arff(Xtest, ytest, vocab, sys.argv[4])
  print_timestamp("finished:")
  
if __name__ == "__main__":
  main()
