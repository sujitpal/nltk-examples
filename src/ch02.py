#! /usr/bin/python
from __future__ import division
import nltk
import nltk.corpus
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

def lexical_diversity(text):
  return len(text) / len(set(text))

def contents(corpus):
  return corpus.fileids()

def describe(corpus):
  print "\t".join(["c/w", "w/s", "w/v", "id"])
  for fileid in corpus.fileids():
    nchars = len(corpus.raw(fileid))
    nwords = len(corpus.words(fileid))
    nsents = len(corpus.sents(fileid))
    nvocab = len(set([w.lower() for w in corpus.words(fileid)]))
    print "\t".join([str(nchars/nwords), str(nwords/nsents),
      str(nwords/nvocab), fileid])

def brown_word_usage_by_category(brown, words):
  for category in brown.categories():
    text = brown.words(categories=category)
    fdist = nltk.FreqDist([w.lower() for w in text])
    print category,
    for word in words:
      print word + ":" + str(fdist[word]),
    print

def brown_word_usage_by_category_cfg(brown, words):
  genres = brown.categories()
  cfd = nltk.ConditionalFreqDist(
    (genre, word.lower())
    for genre in genres
    for word in brown.words(categories=genre))
  cfd.tabulate(conditions=genres, samples=words)

def inaugural_word_usage_by_year(inaugural, words):
  cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in words
    if w.lower().startswith(target))
  cfd.plot()

def udhr_language_length(udhr, languages):
  cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + "-Latin1"))
  cfd.plot(cumulative=True)

def load_local(corpus_root):
  return PlaintextCorpusReader(corpus_root, ".*")

def generate_model(cfdist, start_word, num=15):
  for i in range(num):
    print start_word,
    start_word = cfdist[start_word].max()

def unusual_words(text):
  text_vocab = set(w.lower() for w in text if w.isalpha())
  english_vocab = set(w.lower() for w in nltk.corpus.words.words())
  unusual = text_vocab.difference(english_vocab)
  return sorted(unusual)

def non_stopword_content_pct(text):
  stopwords = nltk.corpus.stopwords.words("english")
  content = [w for w in text if w.lower() not in stopwords]
  return len(content) * 100 / len(text)

def gen_words_puzzle(puzzle_letters, obligatory_letter, min_word_size):
  wordlist = nltk.corpus.words.words()
  plfd = nltk.FreqDist(puzzle_letters)
  return [w for w in wordlist if len(w) >= min_word_size
    and obligatory_letter in w
    and nltk.FreqDist(w) < plfd]

def gender_ambig_names():
  names = nltk.corpus.names
  male_names = names.words("male.txt")
  female_names = names.words("female.txt")
  return [w for w in male_names if w in female_names]

def gender_names_by_firstchar():
  names = nltk.corpus.names
  cfd = nltk.ConditionalFreqDist(
    (fileid, name[0:1])
    for fileid in names.fileids()
    for name in names.words(fileid))
  cfd.plot()

def semantic_similarity(left, right):
  lch = left.lowest_common_hypernyms(right)
  return map(lambda x : x.min_depth(), lch)

if __name__ == "__main__":

#  from nltk.corpus import gutenberg
#  print lexical_diversity(gutenberg.words("austen-emma.txt"))
#  describe(gutenberg)

#  from nltk.corpus import brown
#  modals = ["can", "could", "may", "might", "must", "will"]
#  brown_word_usage_by_category(brown, modals)
#  whwords = ["what", "when", "where", "who", "why"]
#  brown_word_usage_by_category(brown, whwords)
#  modals = ["can", "could", "may", "might", "must", "will"]
#  brown_word_usage_by_category_cfg(brown, modals)

#  from nltk.corpus import inaugural
#  inaugural_word_usage_by_year(inaugural, ["america", "citizen"])

#  from nltk.corpus import udhr
#  languages = ["English", "French_Francais", "German_Deutsch"]
#  udhr_language_length(udhr, languages)
#  raw_text = udhr.raw("English-Latin1")
#  nltk.FreqDist(raw_text).plot()

#  localCorpus = load_local("/usr/share/dict")
#  print localCorpus.fileids()
#  print localCorpus.words("connectives")

#  from nltk.corpus import brown
#  days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
#  genres = ["news", "romance"]
#  cfd = nltk.ConditionalFreqDist(
#    (genre, day)
#    for genre in genres
#    for day in days
#    for word in brown.words(categories=genre) if word.lower() == day)
#  cfd.tabulate(conditions=genres, samples=days)
#  cfd.plot()

#  text = nltk.corpus.genesis.words("english-kjv.txt")
#  bigrams = nltk.bigrams(text)
#  cfd = nltk.ConditionalFreqDist(bigrams)
#  print cfd["living"]
#  generate_model(cfd, "living")

#  # doesn't work (check why not)
#  from com.mycompany.Foo import *
#  com.mycompany.Foo.bar()

#  print unusual_words(nltk.corpus.gutenberg.words("austen-sense.txt"))
#  print unusual_words(nltk.corpus.nps_chat.words())

#  print non_stopword_content_pct(nltk.corpus.reuters.words())

#  print gen_words_puzzle("egivrvonl", "r", 6)

#  print gender_ambig_names()
#  gender_names_by_firstchar()

#  entries = nltk.corpus.cmudict.entries()
#  print len(entries)
#  for word, pron in entries:
#    if len(pron) == 3:
#      ph1, ph2, ph3 = pron
#      if ph1 == "P" and ph3 == "T":
#        print word, pron

#  from nltk.corpus import swadesh
#  print swadesh.fileids()
#  print swadesh.words("en")
#  fr2en = swadesh.entries(["fr", "en"])
#  translate = dict(fr2en)
#  print translate["chien"]
#  print translate["jeter"]
#  de2en = swadesh.entries(["de", "en"])
#  es2en = swadesh.entries(["es", "en"])
#  translate.update(de2en)
#  translate.update(es2en)
#  print translate["Hund"]
#  print translate["perro"]
#
#  languages = ["en", "de", "nl", "es", "fr", "pt", "la"]
#  for i in range(139, 142):
#    print swadesh.entries(languages)[i]

  from nltk.corpus import wordnet as wn
#  print wn.synsets("motorcar")
#  print wn.synset("car.n.01").lemma_names
#  print wn.synset("car.n.01").definition
#  print wn.synset("car.n.01").examples
#  lemmas = wn.synset("car.n.01").lemmas
#  print "lemmas=", lemmas
#  print "synsets(car.n.01.automobile)=", wn.lemma("car.n.01.automobile").synset
#  print "names(car.n.01.automobile)=", wn.lemma("car.n.01.automobile").name
#  print wn.synsets("car")
#  for synset in wn.synsets("car"):
#    print synset.lemma_names
#  print wn.lemmas("car")


#  motorcar = wn.synset("car.n.01")
#  types_of_motorcar = motorcar.hyponyms()
#  print types_of_motorcar[26]
#  print sorted([lemma.name for synset in types_of_motorcar for lemma in synset.lemmas])
#  print motorcar.hypernyms()
#  paths = motorcar.hypernym_paths()
#  print len(paths)
#  print "dist1=", [synset.name for synset in paths[0]]
#  print "dist2=", [synset.name for synset in paths[1]]
#  print motorcar.root_hypernyms()

#  print "part_meronyms(tree)=", wn.synset("tree.n.01").part_meronyms()
#  print "substance_meronyms(tree)=", wn.synset("tree.n.01").substance_meronyms()
#  print "member_holonyms(tree)=", wn.synset("tree.n.01").member_holonyms()

#  for synset in wn.synsets("mint", wn.NOUN):
#    print synset.name + ": " + synset.definition
#
#  print "entailments(walk.v.01)=", wn.synset("walk.v.01").entailments()
#  print "entailments(eat.v.01)=", wn.synset("eat.v.01").entailments()
#  print "entailments(swallow.v.01)=", wn.synset("swallow.v.01").entailments()
#  print "entailments(tease.v.03)=", wn.synset("tease.v.03").entailments()

#  print "antonym(supply.n.02.supply)=", wn.lemma("supply.n.02.supply").antonyms()
#  print dir(wn.synset("harmony.n.02"))

  #Semantic Similarity
  orca = wn.synset("orca.n.01")
  minke = wn.synset("minke_whale.n.01")
  tortoise = wn.synset("tortoise.n.01")
  novel = wn.synset("novel.n.01")
  print "sim(orca,minke)=", semantic_similarity(orca, minke)
  print "sim(orca,tortoise)=", semantic_similarity(orca, tortoise)
  print "sim(orca,novel)=", semantic_similarity(orca, novel)
  print "psim(orca,minke)=", orca.path_similarity(minke)
  print "psim(orca,tortoise)=", orca.path_similarity(tortoise)
  print "psim(orca,novel)=", orca.path_similarity(novel)

  print "end"