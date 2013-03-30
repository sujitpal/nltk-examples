from __future__ import division

import itertools
import sys

import collections
import nltk
import nltk.probability
import numpy as np

######################## utility methods ###########################

def findRareWords(train_file):
  """
  Extra pass through the training file to identify rare
  words and deal with them in the accumulation process.
  """
  wordsFD = nltk.FreqDist()
  reader = nltk.corpus.reader.TaggedCorpusReader(".", train_file)
  for word in reader.words():
    wordsFD.inc(word.lower())
  return set(filter(lambda x: wordsFD[x] < 5, wordsFD.keys()))

def normalizeRareWord(word, rareWords, replaceRare):
  """
  Introduce a bit of variety. Even though they are all rare,
  we classify rare words into "kinds" of rare words.
  """
  if word in rareWords:
    if replaceRare:
      if word.isalnum():
        return "_RARE_NUMERIC_"
      elif word.upper() == word:
        return "_RARE_ALLCAPS_"
      elif word[-1:].isupper():
        return "_RARE_LASTCAP"
      else:
        return "_RARE_"
    else:
      return "_RARE_"
  else:
    return word

def pad(sent, tags=True):
  """
  Pad sentences with the start and stop tags and return padded
  sentence.
  """
  if tags:
    padded = [("<*>", "<*>"), ("<*>", "<*>")]
  else:
    padded = ["<*>", "<*>"]
  padded.extend(sent)
  if tags:
    padded.append(("<$>", "<$>"))
  else:
    padded.append("<$>")
  return padded

def calculateMetrics(actual, predicted):
  """
  Returns the number of cases where prediction and actual NER
  tags are the same, divided by the number of tags for the
  sentence.
  """
  pred_p = map(lambda x: "I" if x == "I" else "O", predicted)
  cm = nltk.metrics.confusionmatrix.ConfusionMatrix(actual, pred_p)
  keys = ["I", "O"]
  metrics = np.matrix(np.zeros((2, 2)))
  for x, y in [(x, y) for x in range(0, 2)
                      for y in range(0, 2)]:
    try:
      metrics[x, y] = cm[keys[x], keys[y]]
    except KeyError:
      pass
  tp = metrics[0, 0]
  tn = metrics[1, 1]
  fp = metrics[0, 1]
  fn = metrics[1, 0]
  precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
  recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
  fmeasure = (0 if (precision + recall) == 0
    else (2 * precision * recall) / (precision + recall))
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  return (precision, recall, fmeasure, accuracy)

def writeResult(fout, hmm, words):
  """
  Writes out the result in the required format.
  """
  tags = hmm.best_path(words)
  for word, tag in zip(words, tags)[2:-1]:
    fout.write("%s %s\n" % (word, tag))
  fout.write("\n")

def bigramToUnigram(bigrams):
  """
  Convert a list of bigrams to the equivalent unigram list.
  """
  unigrams = [bigrams[0][0]]
  unigrams.extend([x[1] for x in bigrams])
  return unigrams

def calculateBackoffTransCPD(tagsFD, transCFD, trans2CFD):
  """
  Uses a backoff model to calculate a smoothed conditional
  probability distribution on the training data.
  """
  probDistDict = collections.defaultdict(nltk.DictionaryProbDist)
  tags = tagsFD.keys()
  conds = [x for x in itertools.permutations(tags, 2)]
  for tag in tags:
    conds.append((tag, tag))
  for (t1, t2) in conds:
    probDict = collections.defaultdict(float)
    prob = 0
    for t3 in tags:
      trigramsFD = trans2CFD[(t1, t2)]
      if trigramsFD.N() > 0 and trigramsFD.freq(t3) > 0:
        prob = trigramsFD.freq(t3) / trigramsFD.N()
      else:
        bigramsFD = transCFD[t2]
        if bigramsFD.N() > 0 and bigramsFD.freq(t3) > 0:
          prob = bigramsFD.freq(t3) / bigramsFD.N()
        else:
          prob = tagsFD[t3] / tagsFD.N()
      probDict[t3] = prob
    probDistDict[(t1, t2)] = nltk.DictionaryProbDist(probDict)
  return nltk.DictionaryConditionalProbDist(probDistDict)

class Accumulator:
  """
  Convenience class to accumulate all the frequencies
  into a set of data structures.
  """
  def __init__(self, rareWords, replaceRare, useTrigrams):
    self.rareWords = rareWords
    self.replaceRare = replaceRare
    self.useTrigrams = useTrigrams
    self.words = set()
    self.tags = set()
    self.priorsFD = nltk.FreqDist()
    self.transitionsCFD = nltk.ConditionalFreqDist()
    self.outputsCFD = nltk.ConditionalFreqDist()
    # additional data structures for trigram
    self.transitions2CFD = nltk.ConditionalFreqDist()
    self.tagsFD = nltk.FreqDist()

  def addSentence(self, sent, norm_func):
    # preprocess
    unigrams = [(norm_func(word, self.rareWords, self.replaceRare), tag)
      for (word, tag) in sent]
    prevTag = None
    prev2Tag = None
    if self.useTrigrams:
      # each state is represented by a tag bigram
      bigrams = nltk.bigrams(unigrams)
      for ((w1, t1), (w2, t2)) in bigrams:
        self.words.add((w1, w2))
        self.tags.add((t1, t2))
        self.priorsFD.inc((t1, t2))
        self.outputsCFD[(t1, t2)].inc((w1, w2))
        if prevTag is not None:
          self.transitionsCFD[prevTag].inc(t2)
        if prev2Tag is not None:
          self.transitions2CFD[prev2Tag].inc((t1, t2))
        prevTag = t2
        prev2Tag = (t1, t2)
        self.tagsFD.inc(prevTag)
    else:
      # each state is represented by an tag unigram
      for word, tag in unigrams:
        self.words.add(word)
        self.tags.add(tag)
        self.priorsFD.inc(tag)
        self.outputsCFD[tag].inc(word)
        if prevTag is not None:
          self.transitionsCFD[prevTag].inc(tag)
        prevTag = tag


####################### train, validate, test ##################

def train(train_file, 
    rareWords, replaceRare, useTrigrams, trigramBackoff):
  """
  Read the file and populate the various frequency and
  conditional frequency distributions and build the HMM
  off these data structures.
  """
  acc = Accumulator(rareWords, replaceRare, useTrigrams)
  reader = nltk.corpus.reader.TaggedCorpusReader(".", train_file)
  for sent in reader.tagged_sents():
    unigrams = pad(sent)
    acc.addSentence(unigrams, normalizeRareWord)
  if useTrigrams:
    if trigramBackoff:
      backoffCPD = calculateBackoffTransCPD(acc.tagsFD, acc.transitionsCFD,
        acc.transitions2CFD)
      return nltk.HiddenMarkovModelTagger(list(acc.words), list(acc.tags),
        backoffCPD,
        nltk.ConditionalProbDist(acc.outputsCFD, nltk.ELEProbDist),
        nltk.ELEProbDist(acc.priorsFD))
    else:
      return nltk.HiddenMarkovModelTagger(list(acc.words), list(acc.tags),
        nltk.ConditionalProbDist(acc.transitions2CFD, nltk.ELEProbDist,
        len(acc.transitions2CFD.conditions())),
        nltk.ConditionalProbDist(acc.outputsCFD, nltk.ELEProbDist),
        nltk.ELEProbDist(acc.priorsFD))
  else:
    return nltk.HiddenMarkovModelTagger(list(acc.words), list(acc.tags),
      nltk.ConditionalProbDist(acc.transitionsCFD, nltk.ELEProbDist,
      len(acc.transitionsCFD.conditions())),
      nltk.ConditionalProbDist(acc.outputsCFD, nltk.ELEProbDist),
      nltk.ELEProbDist(acc.priorsFD))

def validate(hmm, validation_file, rareWords, replaceRare, useTrigrams):
  """
  Tests the HMM against the validation file.
  """
  precision = 0
  recall = 0
  fmeasure = 0
  accuracy = 0
  nSents = 0
  reader = nltk.corpus.reader.TaggedCorpusReader(".", validation_file)
  for sent in reader.tagged_sents():
    sent = pad(sent)
    words = [word for (word, tag) in sent]
    tags = [tag for (word, tag) in sent]
    normWords = map(lambda x: normalizeRareWord(
      x, rareWords, replaceRare), words)
    if useTrigrams:
      # convert words to word bigrams
      normWords = nltk.bigrams(normWords)
    predictedTags = hmm.best_path(normWords)
    if useTrigrams:
      # convert tag bigrams back to unigrams
      predictedTags = bigramToUnigram(predictedTags)
    (p, r, f, a) = calculateMetrics(tags[2:-1], predictedTags[2:-1])
    precision += p
    recall += r
    fmeasure += f
    accuracy += a
    nSents += 1
  print("Accuracy=%f, Precision=%f, Recall=%f, F1-Measure=%f\n" %
    (accuracy/nSents, precision/nSents, recall/nSents,
    fmeasure/nSents))

def test(hmm, test_file, result_file, rareWords, replaceRare, useTrigrams):
  """
  Tests the HMM against the test file (without tags) and writes
  out the results to the result file.
  """
  fin = open(test_file, 'rb')
  fout = open(result_file, 'wb')
  for line in fin:
    line = line.strip()
    words = pad([word for word in line.split(" ")], tags=False)
    normWords = map(lambda x: normalizeRareWord(
      x, rareWords, replaceRare), words)
    if useTrigrams:
      # convert words to word bigrams
      normWords = nltk.bigrams(normWords)
    tags = hmm.best_path(normWords)
    if useTrigrams:
      # convert tag bigrams back to unigrams
      tags = bigramToUnigram(tags)
    fout.write(" ".join(["/".join([word, tag])
      for (word, tag) in (zip(words, tags))[2:-1]]) + "\n")
  fin.close()
  fout.close()

def main():
  normalizeRare = False
  replaceRare = False
  useTrigrams = False
  trigramBackoff = False
  if len(sys.argv) > 1:
    args = sys.argv[1:]
    for arg in args:
      k, v = arg.split("=")
      if k == "normalize-rare":
        normalizeRare = True if v.lower() == "true" else False
      elif k == "replace-rare":
        normalizeRare = True
        replaceRare = True if v.lower() == "true" else False
      elif k == "use-trigrams":
        normalizeRare = True
        replaceRare = True
        useTrigrams = True if v.lower() == "true" else False
      elif k == "trigram-backoff":
        normalizeRare = True
        replaceRare = True
        useTrigrams = True
        trigramBackoff = True if v.lower() == "true" else False
      else:
        continue
  rareWords = set()
  if normalizeRare:
    rareWords = findRareWords("gene.train")
  hmm = train("gene.train",
    rareWords, replaceRare, useTrigrams, trigramBackoff)
  validate(hmm, "gene.validate", rareWords, replaceRare, useTrigrams)
  test(hmm, "gene.test", "gene.test.out",
    rareWords, replaceRare, useTrigrams)
  
if __name__ == "__main__":
  main()
