#!/usr/bin/python

from __future__ import division
import nltk
import re

def bylen(x, y):
  return len(x) - len(y)

def ch04_10_sort_words_by_length(words):
  return sorted(words, cmp=bylen)

def gematrix_score(word):
  if word.isalpha():
    letter_vals = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':80, 'g':3, 'h':8,
      'i':10, 'j':10, 'k':20, 'l':30, 'm':40, 'n':50, 'o':70, 'p':80, 'q':100,
      'r':200, 's':300, 't':400, 'u':6, 'v':6, 'w':800, 'x':60, 'y':10, 'z':7}
    return sum(map(lambda x : letter_vals[x], [c for c in word.lower()]))
  else:
    return 0

def ch04_16_gematria_for_word():
  state_union = nltk.corpus.state_union
  for fileid in state_union.fileids():
    words = state_union.words(fileid)
    satanic_words = filter(lambda x : gematrix_score(x) == 666, words)
    if satanic_words > 0:
      print fileid, len(satanic_words)

def ch04_17_shorten(words, n):
  fd = nltk.FreqDist(words)
  topterms = set()
  topterms.update(fd.keys()[0:n])
  shortened = filter(lambda x : x not in topterms, words)
  return " ".join(shortened)

def ch04_19_sort_by_path_sim(synsets, ref_synset):
  def by_pathsimilarity(x, y):
    diff = ref_synset.path_similarity(x) - ref_synset.path_similarity(y)
    if diff == 0:
      return 0
    elif diff < 0:
      return -1
    else:
      return 1
  return sorted(synsets, cmp=by_pathsimilarity, reverse=True)

def insert_trie(keys):
  trie = nltk.defaultdict()
  [insert_trie_r(trie, key + "_") for key in keys]
  return trie
    
def insert_trie_r(trie, key):
  if len(key) > 1:
    first, rest = key[0], key[1:]
    if first not in trie:
      trie[first] = {}
    insert_trie_r(trie[first], rest)
  else:
    trie[key] = {}

def lookup_trie(trie, key):
  buf = []
  return lookup_trie_r(trie, key + "_", buf)

def lookup_trie_r(trie, key, buf):
  if len(key) > 1:
    first, rest = key[0], key[1:]
    if first not in trie:
      return None
    else:
      buf.append(first)
      return lookup_trie_r(trie[first], rest, buf)
  else:
    if key not in trie:
      return None
    else:
      return "".join(buf)

def ch04_23_lookup_trie():
  trie = insert_trie(["van", "vanity", "vanguard"])
  print lookup_trie(trie, "van")
  print lookup_trie(trie, "vanguard")
  print lookup_trie(trie, "fidelity")

#def ch04_24_keyword_linkage():
#  print "TODO"

def catalan1(n):
  if n == 0 or n == 1:
    return 1
  else:
    return sum([catalan1(i) * catalan1(n - i - 1) for i in range(0,n)])

def catalan2(cache, n):
  if n == 0 or n == 1:
    return 1
  try:
    return cache[n]
  except KeyError:
    cache[n] = sum([catalan1(i) * catalan1(n - i - 1) for i in range(0,n)])
    return cache[n]

def ch04_26_catalan_numbers():
  import time
  cache = {}
  for i in range(0, 10):
    s1 = time.clock()
    cat1 = catalan1(i)
    s1 = time.clock() - s1
    s2 = time.clock()
    cat2 = catalan2(cache, i)
    s2 = time.clock() - s2
    print i, cat1, cat2, s1, s2

#def ch04_27_author_identification():
#  print "TODO"
#
#def ch04_28_gender_lexical_choice():
#  print "TODO"
#
#def ch04_30_uniqueness_point_cutoff():
#  print "TODO"
#
#def ch04_32_summarizer():
#  print "TODO"
#
#def ch04_semantic_orientation_adjectives():
#  print "TODO"
#
#def ch04_statistically_improbable_phrases():
#  print "TODO"

def main():
#  print ch04_10_sort_words_by_length(
#    ["She", "sells", "sea", "shells", "by", "the", "seashore"])

#  print ch04_16_gematria_for_word()

#  print ch04_17_shorten(nltk.corpus.state_union.words("2000-Clinton.txt"), 20)

#  from nltk.corpus import wordnet as wn
#  print ch04_19_sort_by_path_sim(
#    [wn.synset("minke_whale.n.01"), wn.synset("orca.n.01"),
#    wn.synset("novel.n.01"), wn.synset("tortoise.n.01")],
#    wn.synset("right_whale.n.01"))

#  ch04_23_lookup_trie()

  ch04_26_catalan_numbers()
  

if __name__ == "__main__":
  main()
