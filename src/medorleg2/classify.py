#! /usr/bin/python
from __future__ import division

import sys

import cPickle as pickle
import datetime
import nltk
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# total number of sentences (combined)
NTOTAL = 1788280

def calc_ngrams(line):
  """ Converts line into a list of trigram tokens """
  words = nltk.word_tokenize(line.lower())
  word_str = " ".join(words)
  bigrams = nltk.bigrams(words)
  bigram_str = " ".join(["0".join(bigram) for bigram in bigrams])
  trigrams = nltk.trigrams(words)
  trigram_str = " ".join(["0".join(trigram) for trigram in trigrams])
  return " ".join([word_str, bigram_str, trigram_str])
  
def generate_xy(texts, labels):
  ftext = open(texts, 'rb')
  pipeline = Pipeline([
    ("count", CountVectorizer(stop_words='english', min_df=0.0,
#              max_features=10000,
              binary=False)),
    ("tfidf", TfidfTransformer(norm="l2"))
  ])
#  X = pipeline.fit_transform(map(lambda line: calc_ngrams(line), ftext))
  X = pipeline.fit_transform(ftext)
  ftext.close()
  flabel = open(labels, 'rb')
  y = np.loadtxt(flabel)
  flabel.close()
  return X, y

def crossvalidate_model(X, y, nfolds):
  kfold = KFold(X.shape[0], n_folds=nfolds)
  avg_accuracy = 0
  for train, test in kfold:
    Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
    clf = LinearSVC()
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    accuracy = accuracy_score(ytest, ypred)
    print "...accuracy = ", accuracy
    avg_accuracy += accuracy
  print "Average Accuracy: ", (avg_accuracy / nfolds)

def train_model(X, y, binmodel):
  model = LinearSVC()
  model.fit(X, y)
  # reports
  ypred = model.predict(X)
  print "Confusion Matrix (Train):"
  print confusion_matrix(y, ypred)
  print "Classification Report (Train)"
  print classification_report(y, ypred)
  pickle.dump(model, open(binmodel, 'wb'))

def test_model(X, y, binmodel):
  model = pickle.load(open(binmodel, 'rb'))
  if y is not None:
    # reports
    ypred = model.predict(X)
    print "Confusion Matrix (Test)"
    print confusion_matrix(y, ypred)
    print "Classification Report (Test)"
    print classification_report(y, ypred)

def print_timestamp(message):
  print message, datetime.datetime.now()

def usage():
  print "Usage: python classify.py [xval|test|train]"
  sys.exit(-1)
  
def main():
  if len(sys.argv) != 2:
    usage()
  print_timestamp("started:")
  X, y = generate_xy("data/sentences.txt", "data/labels.txt")
  if sys.argv[1] == "xval":
    crossvalidate_model(X, y, 10)
  elif sys.argv[1] == "run":
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
      test_size=0.1, random_state=42)
    train_model(Xtrain, ytrain, "data/model.bin")
    test_model(Xtest, ytest, "data/model.bin")
  else:
    usage()
  print_timestamp("finished:")
  
if __name__ == "__main__":
  main()
