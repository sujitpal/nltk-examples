#!/usr/bin/python

import sys
import cPickle as pickle
from cener_lib import *
from nltk.tokenize import sent_tokenize, word_tokenize

def train_ner(pickle_file):
  # initialize
  pos_tagger = train_pos_tagger()
  ceps = ce_phrases()
  cep_words = ce_phrase_words(ceps)
  # train classifier
  sentfile = open("cnet_reviews_sents.txt", 'rb')
  featuresets = []
  for sent in sentfile:
    tagged_sent = tag(sent, pos_tagger, ceps, cep_words)
    for idx, (word, pos_tag, io_tag) in enumerate(tagged_sent):
      featuresets.append((word_features(tagged_sent, idx), io_tag))
  sentfile.close()
  split = int(0.9 * len(featuresets))
#  random.shuffle(featuresets)
  train_set, test_set = featuresets[0:split], featuresets[split:]
#  classifier = nltk.NaiveBayesClassifier.train(train_set)
#  classifier = nltk.DecisionTreeClassifier.train(train_set)
  classifier = nltk.MaxentClassifier.train(train_set, algorithm="GIS", trace=0)
  # evaluate classifier
  print "accuracy=", nltk.classify.accuracy(classifier, test_set)
  if pickle_file != None:
    # pickle classifier
    pickled_classifier = open(pickle_file, 'wb')
    pickle.dump(classifier, pickled_classifier)
    pickled_classifier.close()
  return classifier

def get_trained_ner(pickle_file):
  pickled_classifier = open(pickle_file, 'rb')
  classifier = pickle.load(pickled_classifier)
  pickled_classifier.close()
  return classifier

def test_ner(input_file, classifier):
  pos_tagger = train_pos_tagger()
  input = open(input_file, 'rb')
  for line in input:
    line = line[:-1]
    if len(line.strip()) == 0:
      continue
    for sent in sent_tokenize(line):
      tokens = word_tokenize(sent)
      pos_tagged = pos_tagger.tag(tokens)
      io_tags = []
      for idx, (word, pos) in enumerate(pos_tagged):
        io_tags.append(classifier.classify(word_features(pos_tagged, idx)))
      ner_sent = zip(tokens, io_tags)
      print_sent = []
      for token, io_tag in ner_sent:
        if io_tag == True:
          print_sent.append("<u>" + token + "</u>")
        else:
          print_sent.append(token)
      print " ".join(print_sent)

  input.close()
      
def main():
  if len(sys.argv) != 2:
    print "Usage ./cener.py [train|test]"
    sys.exit(-1)
  if sys.argv[1] == "train":
    classifier = train_ner("ce_ner_classifier.pkl")
  else:
    classifier = get_trained_ner("ce_ner_classifier.pkl")
    test_ner("test.txt", classifier)
  
if __name__ == "__main__":
  main()
