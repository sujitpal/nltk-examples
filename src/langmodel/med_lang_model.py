from __future__ import division

import math
import os.path

import cPickle
import glob
import nltk
from nltk.corpus.reader import XMLCorpusReader

class LangModel:
  def __init__(self, order, alpha, sentences):
    self.order = order
    self.alpha = alpha
    if order > 1:
      self.backoff = LangModel(order - 1, alpha, sentences)
      self.lexicon = None
    else:
      self.backoff = None
      self.lexicon = set()
    self.ngramFD = nltk.FreqDist()
    for sentence in sentences:
      words = nltk.word_tokenize(sentence)
      wordNGrams = nltk.ngrams(words, order)
      for wordNGram in wordNGrams:
        self.ngramFD.inc(wordNGram)
        if order == 1:
          self.lexicon.add(wordNGram)

  def logprob(self, ngram):
    return math.log(self.prob(ngram))
  
  def prob(self, ngram):
    if self.backoff != None:
      freq = self.ngramFD[ngram]
      backoffFreq = self.backoff.ngramFD[ngram[1:]]
      if freq == 0:
        return self.alpha * self.backoff.prob(ngram[1:])
      else:
        return freq / backoffFreq
    else:
      return 1 / len(self.lexicon)

def train():
  if os.path.isfile("lm.bin"):
    return
  files = glob.glob("data/*.xml")
  sentences = []
  i = 0
  for file in files:
    if i > 0 and i % 500 == 0:
      print("%d/%d files loaded, #-sentences: %d" %
        (i, len(files), len(sentences)))
    dir, file = file.split("/")
    reader = XMLCorpusReader(dir, file)
    sentences.extend(nltk.sent_tokenize(" ".join(reader.words())))
    i += 1
  lm = LangModel(3, 0.4, sentences)
  cPickle.dump(lm, open("lm.bin", "wb"))

def test():
  lm1 = cPickle.load(open("lm.bin", 'rb'))
  testFile = open("sentences.test", 'rb')
  for line in testFile:
    sentence = line.strip()
    print "SENTENCE:", sentence,
    words = nltk.word_tokenize(sentence)
    wordTrigrams = nltk.trigrams(words)
    slogprob = 0
    for wordTrigram in wordTrigrams:
      logprob = lm1.logprob(wordTrigram)
      slogprob += logprob
    print slogprob / len(words)

def main():
  train()
  test()

if __name__ == "__main__":
  main()