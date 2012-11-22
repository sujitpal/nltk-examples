#!/usr/bin/python

from __future__ import division
import nltk
import re

def ch03_10():
  sent = re.split(" ", "The dog gave John the newspaper")
  print [(w, len(w)) for w in sent]

def ch03_18_wh_words():
  moby_dick = nltk.corpus.gutenberg.words("melville-moby_dick.txt")
  print [w for w in moby_dick if w.startswith("wh")]

def ch03_29_reading_difficulty():
  sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
  from nltk.corpus import brown
  for category in brown.categories():
    raw = brown.raw(categories=category)
    words = len(brown.words(categories=category))
    sentences = len(sent_tokenizer.tokenize(raw))
    letters_per_word = (len(raw) - words) / words # raw chars - words space chars
    words_per_sentence = words / sentences
    reading_level = (4.71 * letters_per_word) + (0.5 * words_per_sentence) + 21.43
    print category, reading_level

def ch03_30_porter_vs_lancaster():
  porter = nltk.PorterStemmer()
  lancaster = nltk.LancasterStemmer()
  tokens = ["When", "all", "is", "said", "and", "done", ",", "more", "is", "said", "than", "done", "."]
  print "porter=", [porter.stem(w.lower()) for w in tokens]
  print "lancaster=", [lancaster.stem(w.lower()) for w in tokens]
  print "len(tokens)=", map(lambda token : len(token), tokens)

def ch03_42_wordnet_semantic_index():
  from nltk.corpus import webtext
  from nltk.corpus import wordnet as wn
  postings = []
  docids = {}
  for (pos, fileid) in enumerate(webtext.fileids()):
    docids[pos] = fileid
    wpos = 0
    words = webtext.words(fileid)
    for word in words:
      try:
        postings.append((word.lower(), (pos, wpos)))
        offset = wn.synsets(word)[0].offset
        postings.append((offset, (pos, wpos)))
        poffset = wn.synsets(word)[0].hypernyms()[0].offset
        postings.append((poffset, (pos, wpos)))
      except IndexError:
        continue
      wpos = wpos + 1
  index = nltk.Index(postings)
  query = "canine"
  qpostings = []
  qpostings.extend([(pos, wpos) for (pos, wpos) in index[query]])
  try:
    offset = wn.synsets(query)[0].offset
    qpostings.extend([(pos, wpos) for (pos, wpos) in index[offset]])
  except IndexError:
    pass
  for (pos, wpos) in qpostings:
    left = webtext.words(docids[pos])[wpos-4:wpos]
    right = webtext.words(docids[pos])[wpos:wpos+4]
    print left, right

def bigram_freqdist(words):
  return nltk.FreqDist(["".join(w)
    for word in words
    for w in nltk.bigrams(word.lower())])
    
def ch03_43_translate():
  from nltk.corpus import udhr
  en_fd = bigram_freqdist(udhr.words("English-Latin1"))
  fr_fd = bigram_freqdist(udhr.words("French_Francais-Latin1"))
  de_fd = bigram_freqdist(udhr.words("German_Deutsch-Latin1"))
  es_fd = bigram_freqdist(udhr.words("Spanish-Latin1"))
  inputs = ["Nice day", "Guten Tag", "Buenas Dias", "Tres Bien"]
  for input in inputs:
    words = input.lower().split(" ")
    # TODO: remove keys present in reference set
    ranks = map(lambda x : nltk.spearman_correlation(x, bigram_freqdist(words)),
      [en_fd, fr_fd, de_fd, es_fd])
    print input, ranks
    
def main():
#  ch03_10()
#  ch03_18_wh_words()
#  ch03_29_reading_difficulty()
#  ch03_30_porter_vs_lancaster()
  ch03_42_wordnet_semantic_index()
#  ch03_43_translate()

if __name__ == "__main__":
  main()
