from __future__ import division
import operator
import nltk
import numpy as np
from scipy.stats import binom
import string

def isValid(word):
  if word.startswith("#"):
    return False # no hashtag
  else:
    vword = word.translate(string.maketrans("", ""), string.punctuation)
    return len(vword) == len(word)

def llr(c1, c2, c12, n):
  # H0: Independence p(w1,w2) = p(w1,~w2) = c2/N
  p0 = c2 / n
  # H1: Dependence, p(w1,w2) = c12/N
  p10 = c12 / n
  # H1: p(~w1,w2) = (c2-c12)/N
  p11 = (c2 - c12) / n
  # binomial probabilities
  # H0: b(c12; c1, p0),  b(c2-c12; N-c1, p0)
  # H1: b(c12, c1, p10), b(c2-c12; N-c1, p11)
  probs = np.matrix([
    [binom(c1, p0).logpmf(c12), binom(n - c1, p0).logpmf(c2 - c12)],
    [binom(c1, p10).logpmf(c12), binom(n - c1, p11).logpmf(c2 - c12)]])
  # LLR = p(H1) / p(H0)
  return np.sum(probs[1, :]) - np.sum(probs[0, :])

def isLikelyNGram(ngram, phrases):
  if len(ngram) == 2:
    return True
  prevGram = ngram[:-1]
  return phrases.has_key(prevGram)

def main():
  # accumulate words and word frequency distributions
  lines = []
  unigramFD = nltk.FreqDist()
  fin = open("twitter_messages.txt", 'rb')
  i = 0
  for line in fin:
    i += 1
    words = nltk.word_tokenize(line.strip().lower())
    words = filter(lambda x: isValid(x), words)
    [unigramFD.inc(x) for x in words]
    lines.append(words)
    if i > 1000:
      break
  fin.close()
  # identify likely phrases using a multi-pass algorithm based
  # on the LLR approach described in the Building Search Applications
  # Lucene, LingPipe and GATE book, except that we treat n-gram
  # collocations beyond 2 as n-1 gram plus a unigram.
  phrases = nltk.defaultdict(float)
  prevGramFD = None
  for i in range(2, 5):
    ngramFD = nltk.FreqDist()
    for words in lines:
      nextGrams = nltk.ngrams(words, i)
      nextGrams = filter(lambda x: isLikelyNGram(x, phrases), nextGrams)
      [ngramFD.inc(x) for x in nextGrams]
    for k, v in ngramFD.iteritems():
      if v > 1:
        c1 = unigramFD[k[0]] if prevGramFD == None else prevGramFD[k[:-1]]
        c2 = unigramFD[k[1]] if prevGramFD == None else unigramFD[k[len(k) - 1]]
        c12 = ngramFD[k]
        n = unigramFD.N() if prevGramFD == None else prevGramFD.N()
        phrases[k] = llr(c1, c2, c12, n)
    # only consider bigrams where LLR > 0, ie P(H1) > P(H0)
    likelyPhrases = nltk.defaultdict(float)
    likelyPhrases.update([(k, v) for (k, v)
      in phrases.iteritems() if len(k) == i and v > 0])
    print "==== #-grams = %d ====" % (i)
    sortedPhrases = sorted(likelyPhrases.items(),
      key=operator.itemgetter(1), reverse=True)
    for k, v in sortedPhrases:
      print k, v
    prevGramFD = ngramFD

if __name__ == "__main__":
  main()