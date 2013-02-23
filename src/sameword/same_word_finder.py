from __future__ import division
from nltk.corpus import wordnet as wn
import sys

def similarity(w1, w2, sim=wn.path_similarity):
  synsets1 = wn.synsets(w1)
  synsets2 = wn.synsets(w2)
  sim_scores = []
  for synset1 in synsets1:
    for synset2 in synsets2:
      sim_scores.append(sim(synset1, synset2))
  if len(sim_scores) == 0:
    return 0
  else:
    return max(sim_scores)

def main():
  f = open(sys.argv[1], 'rb')
  for line in f:
    (word1, word2) = line.strip().split("\t")
    if similarity(word1, word2) != 1.0:
      print word1
  f.close()

if __name__ == "__main__":
  main()
