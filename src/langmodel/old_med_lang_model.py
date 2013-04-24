from __future__ import division

import glob
import nltk
from nltk.corpus.reader import XMLCorpusReader
from nltk.model.ngram import NgramModel
from nltk.probability import LidstoneProbDist
import cPickle

def train():
  # parse XML and load up words
  print("Loading words from XML files...")
  sentences = []
  files = glob.glob("data/*.xml")
  i = 0
  for file in files:
    if i > 0 and i % 500 == 0:
      print("%d/%d files loaded, #-sentences: %d" %
        (i, len(files), len(sentences)))
      break
    dir, file = file.split("/")
    reader = XMLCorpusReader(dir, file)
    sentences.extend(nltk.sent_tokenize(" ".join(reader.words())))
    i += 1
  words = []
  for sentence in sentences:
    words.append(nltk.word_tokenize(sentence))
  # build a trigram Language Model (using default Good-Turing
  # smoothing) with the words array
  print("Building language model...")
  est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
  langModel = NgramModel(3, words, estimator=est)
#  langModel = NgramModel(3, words)
#  cPickle.dump(langModel, open("lm.bin", 'wb'))
  return langModel

def test(langModel):
  testData = open("sentences.test", 'rb')
  for line in testData:
    sentence = line.strip()
    print "SENTENCE:", sentence,
    words = nltk.word_tokenize(sentence)
    trigrams = nltk.trigrams(words)
    slogprob = 0
    for trigram in trigrams:
      word = trigram[2]
      context = list(trigrams[:-1])
      slogprob += langModel.logprob(word, context)
    print "(", slogprob, ")"
  testData.close()

def main():
  langModel = train()
  test(langModel)

if __name__ == "__main__":
  main()