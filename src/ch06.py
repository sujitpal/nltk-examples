#!/usr/bin/python
# Text classification

from __future__ import division
import nltk
from nltk.classify import apply_features
import random
import math

def _gender_features(word):
  features = {}
  # started with 1 feature
  features["last_letter"] = word[-1].lower()
  # added some more
  features["first_letter"] = word[0].lower()
  for letter in "abcdefghijklmnopqrstuvwxyz":
    features["count(%s)" % letter] = word.lower().count(letter)
    features["has(%s)" % letter] = (letter in word.lower())
  # result of error analysis:
  # names ending in -yn are mostly female, and names ending
  # in -ch ar mostly male, so add 2 more features
  features["suffix2"] = word[-2:]
  return features

def naive_bayes_gender_classifier():
  from nltk.corpus import names
  names = ([(name, "male") for name in names.words("male.txt")] +
           [(name, "female") for name in names.words("female.txt")])
  random.shuffle(names)
#  featuresets = [(_gender_features(n), g) for (n,g) in names]
#  train_set, test_set = featuresets[500:], featuresets[:500]
  # advisable to stream the sets in for large data set.
  train_set = apply_features(_gender_features, names[500:])
  test_set = apply_features(_gender_features, names[:500])
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print "Neo is ", classifier.classify(_gender_features("Neo"))
  print "Trinity is", classifier.classify(_gender_features("Trinity"))
  # calculate the accuracy of the classifier
  print nltk.classify.accuracy(classifier, test_set)
  classifier.show_most_informative_features(5)

def error_analysis():
  from nltk.corpus import names
  names = ([(name, "male") for name in names.words("male.txt")] +
           [(name, "female") for name in names.words("female.txt")])
  random.shuffle(names)
  test_names, devtest_names, train_names = \
    names[:500], names[500:1500], names[1500:]
  train_set = [(_gender_features(n), g) for (n,g) in train_names]
  devtest_set = [(_gender_features(n), g) for (n,g) in devtest_names]
  test_set = [(_gender_features(n), g) for (n,g) in test_names]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print nltk.classify.accuracy(classifier, devtest_set)
  errors = []
  for (name, tag) in devtest_names:
    guess = classifier.classify(_gender_features(name))
    if guess != tag:
      errors.append((tag, guess, name))
  for (tag, guess, name) in sorted(errors):
    print "correct=%s, guess=%s, name=%s" % (tag, guess, name)

def _document_features(document, word_features):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains(%s)' % word] = (word in document_words)
  return features
  
def document_classification_movie_reviews():
  from nltk.corpus import movie_reviews
  documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
  random.shuffle(documents)
  # use the most frequest 2000 words as features
  all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
  word_features = all_words.keys()[:2000]
  featuresets = [(_document_features(d, word_features), category)
                 for (d,category) in documents]
  train_set, test_set = featuresets[100:], featuresets[:100]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print nltk.classify.accuracy(classifier, test_set)
  classifier.show_most_informative_features(30)

def _pos_features(word, common_suffixes):
  features = {}
  for suffix in common_suffixes:
    features["endswith(%s)" % suffix] = word.lower().endswith(suffix)
  return features

def pos_tagging_classification():
  # find most common suffixes of words
  from nltk.corpus import brown
  suffix_fdist = nltk.FreqDist()
  for word in brown.words():
    word = word.lower()
    suffix_fdist.inc(word[-1:])
    suffix_fdist.inc(word[-2:])
    suffix_fdist.inc(word[-3:])
  common_suffixes = suffix_fdist.keys()[:100]
  tagged_words = brown.tagged_words(categories="news")
  featuresets = [(_pos_features(w, common_suffixes), pos)
                  for (w, pos) in tagged_words]
  size = int(len(featuresets) * 0.1)
  train_set, test_set = featuresets[size:], featuresets[:size]
  classifier = nltk.DecisionTreeClassifier.train(train_set)
  print nltk.classify.accuracy(classifier, test_set)
  print classifier.pseudocode(depth=4)
  
def _pos_features2(sentence, i):
  features = {
    "suffix(1)" : sentence[i][-1:],
    "suffix(2)" : sentence[i][-2:],
    "suffix(3)" : sentence[i][-3:]}
  if i == 0:
    features["prev-word"] = "<START>"
  else:
    features["prev-word"] = sentence[i - 1]
  return features

def pos_tagging_classification_with_sentence_context():
  from nltk.corpus import brown
  tagged_sents = brown.tagged_sents(categories="news")
  featuresets = []
  for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
      featuresets.append((_pos_features2(untagged_sent, i), tag))
  size = int(len(featuresets) * 0.1)
  train_set, test_set = featuresets[size:], featuresets[:size]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print nltk.classify.accuracy(classifier, test_set)

def _pos_features3(sentence, i, history):
  features = {
    "suffix(1)" : sentence[i][-1:],
    "suffix(2)" : sentence[i][-2:],
    "suffix(3)" : sentence[i][-3:]}
  if i == 0:
    features["prev-word"] = "<START>"
    features["prev-tag"] = "<START>"
  else:
    features["prev-word"] = sentence[i-1]
    features["prev-tag"] = history[i-1]
  return features

class ConsecutivePosTagger(nltk.TaggerI):
  def __init__(self, train_sents):
    train_set = []
    for tagged_sent in train_sents:
      untagged_sent = nltk.tag.untag(tagged_sent)
      history = []
      for i, (word, tag) in enumerate(tagged_sent):
        featureset = _pos_features3(untagged_sent, i, history)
        train_set.append((featureset, tag))
        history.append(tag)
    self.classifier = nltk.NaiveBayesClassifier.train(train_set)

  def tag(self, sentence):
    history = []
    for i, word in enumerate(sentence):
      featureset = _pos_features3(sentence, i, history)
      tag = self.classifier.classify(featureset)
      history.append(tag)
    return zip(sentence, history)

def sequence_classification_using_prev_pos():
  from nltk.corpus import brown
  tagged_sents = brown.tagged_sents(categories="news")
  size = int(len(tagged_sents) * 0.1)
  train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
  tagger = ConsecutivePosTagger(train_sents)
  print tagger.evaluate(test_sents)

def _punct_features(tokens, i):
  return {
    "next-word-capitalized" : tokens[i+1][0].isupper(),
    "prevword" : tokens[i-1].lower(),
    "punct" : tokens[i],
    "prev-word-is-one-char" : len(tokens[i-1]) == 1}

def sentence_segmentation_as_classification_for_punctuation():
  from nltk.corpus import treebank_raw
  sents = treebank_raw.sents()
  tokens = []
  boundaries = set()
  offset = 0
  for sent in sents:
    # each sent is a list of words, added flat into sents
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset - 1) # all sentence boundary tokens
  featuresets = [(_punct_features(tokens, i), (i in boundaries))
                 for i in range(1, len(tokens) - 1)
                 if tokens[i] in ".?!"]
  size = int(len(featuresets) * 0.1)
  train_set, test_set = featuresets[size:], featuresets[:size]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print nltk.classify.accuracy(classifier, test_set)

def _dialog_act_features(post):
  features = {}
  for word in nltk.word_tokenize(post):
    features["contains(%s)" % word.lower()] = True
  return features

def identify_dialog_act_types():
  posts = nltk.corpus.nps_chat.xml_posts()[:10000]
  featuresets = [(_dialog_act_features(post.text), post.get("class"))
                 for post in posts]
  size = int(len(featuresets) * 0.1)
  train_set, test_set = featuresets[size:], featuresets[:size]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print nltk.classify.accuracy(classifier, test_set)

def _rte_features(rtepair):
  # builds a bag of words for both text and hypothesis
  # after throwing away some stopwords
  extractor = nltk.RTEFeatureExtractor(rtepair)
  return {
    "word_overlap" : len(extractor.overlap("word")),
    "word_hyp_extra" : len(extractor.hyp_extra("word")),
    "ne_overlap" : len(extractor.overlap("ne")),
    "ne_hyp_overlap" : len(extractor.hyp_extra("ne"))}

def recognize_text_entailment():
  rtepair = nltk.corpus.rte.pairs(["rte3_dev.xml"])[33]
  extractor = nltk.RTEFeatureExtractor(rtepair)
  # all important words in hypothesis is contained in text => entailment
  print "text-words=", extractor.text_words
  print "hyp-words=", extractor.hyp_words
  print "overlap(word)=", extractor.overlap("word")
  print "overlap(ne)=", extractor.overlap("ne")
  print "hyp_extra(word)=", extractor.hyp_extra("word")
  print "hyp_extra(ne)=", extractor.hyp_extra("ne")

def entropy(labels):
  freqdist = nltk.FreqDist(labels)
  probs = [freqdist.freq(label) for label in labels]
  return -sum([p * math.log(p, 2) for p in probs])

def calc_entropy():
  print entropy(["male", "male", "male", "female"])
  print entropy(["male", "male", "male", "male"])
  print entropy(["female", "female", "female", "female"])

def main():
#  naive_bayes_gender_classifier()
#  error_analysis()
  document_classification_movie_reviews()
#  pos_tagging_classification()
#  pos_tagging_classification_with_sentence_context()
#  sequence_classification_using_prev_pos()
#  sentence_segmentation_as_classification_for_punctuation()
#  identify_dialog_act_types()
#  recognize_text_entailment()
#  calc_entropy()
  print "end"


if __name__ == "__main__":
  main()
