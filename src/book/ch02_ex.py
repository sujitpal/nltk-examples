#!/usr/bin/python

from __future__ import division
import nltk
import operator

def ex1():
  phrase = ["This", "is", "the", "house", "that", "Jack", "built", "."]
  print phrase + phrase
#  print phrase - phrase
#  print phrase * phrase
#  print phrase / phrase
  print sorted(phrase)

def ex2():
  from nltk.corpus import gutenberg
  ap = gutenberg.words("austen-persuasion.txt")
  word_tokens = len(ap)
  word_types = len(set([w.lower() for w in ap]))
  print "#-word tokens=", word_tokens
  print "#-word types=", word_types

def ex4():
  from nltk.corpus import state_union
  tags = ["men", "women", "people"]
#  for fileid in state_union.fileids():
#    words = state_union.words(fileid)
#    fdist = nltk.FreqDist([w.lower() for w in words])
#    print fileid + ": ",
#    for tag in tags:
#      print tag + "=" + str(fdist[tag]) + " ",
#    print
  cfd = nltk.ConditionalFreqDist(
    (target, fileid[0:4])
    for fileid in state_union.fileids()
    for w in state_union.words(fileid)
      for target in tags if w.lower() == target)
  cfd.plot()

def ex5():
  from nltk.corpus import wordnet as wn
  for w in ["jaguar", "transistor", "train"]:
    s = wn.synset(w + ".n.01")
    if (s is not None):
      print "member_meronym(" + w + ")=", s.member_meronyms()
      print "part_meronym(" + w + ")=", s.part_meronyms()
      print "substance_meronym(" + w + ")=", s.substance_meronyms()
      print "member_holonym(" + w + ")=", s.member_holonyms()
      print "part_holonym(" + w + ")=", s.part_holonyms()
      print "substance_holonym(" + w + ")=", s.substance_holonyms()

def ex7():
  from nltk.corpus import gutenberg
  for fileid in gutenberg.fileids():
    text = nltk.Text(gutenberg.words(fileid))
    print ("file: " + fileid)
    print text.concordance("however")

def ex8():
  from nltk.corpus import names
  genders = ["male", "female"]
  alphabets = ["A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
  cfd = nltk.ConditionalFreqDist(
    (gender, firstletter)
    for gender in genders
    for firstletter in alphabets
    for letter_count in
    [len(w) for w in names.words(gender + ".txt")
    if w[0:1] == firstletter])
  cfd.plot()

def ex10():
  from nltk.corpus import brown
  from nltk.corpus import stopwords
  stopwords = stopwords.words("english")
  for genre in brown.categories():
    print genre
    words = map(lambda x : x.lower(), brown.words(categories=genre))
    fd = nltk.FreqDist([w for w in words if w.isalpha() and not(w in stopwords)])
    vocab_size = len(set(words))
    sum = 0
    for word in fd.keys():
      freq = fd[word]
      print "... %s (%f)" % (word, (freq * 100 / vocab_size))
      sum = sum + freq
      if (sum > (vocab_size / 3)):
        break

def ex11():
  from nltk.corpus import brown
  modals = set(["can", "could", "may", "might", "shall", "should", "will", "would", "must", "ought"])
  cfd = nltk.ConditionalFreqDist(
    (genre, modal)
    for genre in brown.categories()
    for modal in [w.lower() for w in brown.words(categories=genre) if w.lower() in modals])
  cfd.plot()

def ex12():
  from nltk.corpus import cmudict
  entries = cmudict.entries()
  words = map(lambda (word, pron) : word, entries)
  distinct_words = set(words)
  fd = nltk.FreqDist(words)
  multi_prons = 0
  for key in fd.keys():
    if fd[key] == 1:
      break
    multi_prons = multi_prons + 1
  print "#-distinct words:", len(distinct_words)
  print "#-words with multiple prons:", multi_prons

def ex13():
  from nltk.corpus import wordnet as wn
  num_synsets = 0
  num_synsets_wo_hyponyms = 0
  for noun_synset in wn.all_synsets("n"):
    if len(noun_synset.hyponyms()) == 0:
      num_synsets_wo_hyponyms = num_synsets_wo_hyponyms + 1
    num_synsets = num_synsets + 1
  print num_synsets_wo_hyponyms * 100 / num_synsets

def ex14_supergloss(s):
  from nltk.corpus import wordnet as wn
  ss = wn.synset(s)
  buf = ss.definition[0:1].upper() + ss.definition[1:]
  for hyponym in ss.hyponyms():
    buf = buf + ". " + hyponym.definition[0:1].upper() + hyponym.definition[1:]
  for hypernym in ss.hypernyms():
    buf = buf + ". " + hypernym.definition[0:1].upper() + hypernym.definition[1:]
  print buf

def ex15():
  from nltk.corpus import brown
  fd = nltk.FreqDist([w.lower() for w in brown.words()])
  print filter(lambda k : fd[k] > 3, fd.keys())

def ex16():
  from nltk.corpus import brown
  lex_div = {}
  for category in brown.categories():
    words = brown.words(categories=category)
    ld = len(words) / len(set(words))
    print category, ld
    lex_div[category] = ld
  print sorted(lex_div.iteritems(), key=operator.itemgetter(1))

def ex17():
  from nltk.corpus import gutenberg
  macbeth = gutenberg.words("shakespeare-macbeth.txt")
  stopwords = set(nltk.corpus.stopwords.words())
  fd = nltk.FreqDist([w for w in macbeth if w.lower() not in stopwords
      and len(w) > 3 and w.isalpha()])
  print fd.keys()[0:50]

def ex18():
  from nltk.corpus import gutenberg
  macbeth = gutenberg.words("shakespeare-macbeth.txt")
  stopwords = set(nltk.corpus.stopwords.words())
  bigrams = nltk.bigrams(macbeth)
  print bigrams
  bigrams_wo_stopwords = filter(lambda (k, v) : k not in stopwords
    and v not in stopwords
    and k.isalpha()
    and v.isalpha(), bigrams)
  fd = nltk.FreqDist(map(lambda (k,v) : k+":"+v, bigrams_wo_stopwords))
  print map(lambda k : (k.split(":")[0], k.split(":")[1]), fd.keys())[0:50]

def ex25_findlanguage():
  from nltk.corpus import udhr
  word_lang_map = {}
  for fileid in udhr.fileids():
    if fileid.endswith("-Latin1"):
      lang = fileid[:-7]
      words = udhr.words(fileid)
      for word in words:
        try:
          word_lang_map[word]
        except KeyError:
          word_lang_map[word] = set()
        langs = word_lang_map[word]
        langs.add(lang)
        word_lang_map[word] = langs
  print word_lang_map["arashobora"]

def ex26_branchingfactor():
  from nltk.corpus import wordnet as wn
  num_synsets = 0
  num_hyponyms = 0
  for noun_synset in wn.all_synsets("n"):
    (num_hyponyms, num_synsets) = \
      branchingfactor_r(noun_synset, num_synsets, num_hyponyms)
  print "branching factor=", (num_hyponyms / num_synsets)

def branchingfactor_r(synset, num_synsets, num_hyponyms):
  num_synsets = num_synsets + 1
  for hyponym in synset.hyponyms():
    branchingfactor_r(hyponym, num_synsets, num_hyponyms)
    num_hyponyms = num_hyponyms + 1
  return (num_hyponyms, num_synsets)

def ex27_polysemy():
  from nltk.corpus import wordnet as wn
  for pos in ["n", "v", "a"]:
    synsets = wn.all_synsets(pos)
    num_synsets = 0
    num_senses = 0
    for synset in synsets:
      num_synsets = num_synsets + 1
      num_senses = num_senses + len(synset.lemmas)
    print "polysemy(" + pos + ")=", (num_senses / num_synsets)

def main():
#  ex1()
#  ex2()
#  ex4()
#  ex5()
#  ex7()
#  ex8()
#  ex10()
#  ex11()
#  ex12()
#  ex13()
#  ex14_supergloss("car.n.01")
#  ex15()
#  ex16()
#  ex17()
#  ex18()
#  ex25_findlanguage()
#  ex26_branchingfactor()
  ex27_polysemy()
  
if __name__ == "__main__":
  main()