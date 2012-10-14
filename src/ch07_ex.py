#!/usr/bin/python

from __future__ import division
import nltk
import re

def ch07_02_match_np_containing_plural_nouns():
  grammar = r"""
    NP : {<JJ> <NNS>}
  """
  sent = [("Many", "JJ"), ("researchers", "NNS"), ("two", "CD"),
    ("weeks", "NNS"), ("both", "DT"), ("new", "JJ"), ("positions", "NNS")]
  cp = nltk.RegexpParser(grammar)
  print cp.parse(sent)

def ch07_03_develop_grammar_with_chunkparser():
  # nltk.app.chunkparser()
  from nltk.corpus import conll2000
  grammar = r"""
    NP: {<NN.*>}
       {<DT> <NN> <JJ> <NN>}
       {<DT> <JJ>* <NN.*>}
       {<POS> <JJ>* <NN>}
       {<NNP> <CC> <NNP>}
  """
  cp = nltk.RegexpParser(grammar)
  for sentence in conll2000.chunked_sents("train.txt", chunk_types=["NP"]):
    print cp.parse(sentence)

def ch07_05_tag_pattern_np_containing_gerund():
  grammar = r"""
  NP: {<.*> <VBG> <NN.*>}
  """
  cp = nltk.RegexpParser(grammar)
  from nltk.corpus import brown
  tagged_sents = brown.tagged_sents(categories="news")
  for sent in tagged_sents:
    tree = str(cp.parse(sent))
    if tree.find("(NP ") > -1:
      print tree

def ch07_06_coordinated_noun_phrases():
  from nltk.corpus import brown
  tagged_sents = brown.tagged_sents(categories="news")
  grammar = r"""
    NP_CC: {<NN.*> <CC> <NN.*>}
           {<DT> <PRP> <NN.*> <CC> <NN.*>}
           {<NN.*>+ <CC> <NN.*>}
  """
  cp = nltk.RegexpParser(grammar)
  for sent in tagged_sents:
    tree = str(cp.parse(sent))
    if tree.find("(NP_CC ") > -1:
      print tree

def ch07_07_chunker_eval():
  from nltk.corpus import conll2000
  grammar = r"""
    NP: {<NN.*>}
       {<DT> <NN> <JJ> <NN>}
       {<DT> <JJ>* <NN.*>}
       {<POS> <JJ>* <NN>}
       {<NNP> <CC> <NNP>}
  """
  cp = nltk.RegexpParser(grammar)
  test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
  print cp.evaluate(test_sents)
#  print cp.chunkscore.missed()
#  print cp.chunkscore.incorrect()

def ch07_13a_tag_seqs_for_np():
  from nltk.corpus import conll2000
  train_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
  fdist = nltk.FreqDist()
  tagseq = []
  for sent in train_sents:
    for word, postag, iobtag in nltk.chunk.tree2conlltags(sent):
      if iobtag == "B-NP":
        fdist.inc(" ".join(tagseq))
        tagseq = []
        tagseq.append(postag)
      elif iobtag == "O":
        continue
      else:
        tagseq.append(postag)
  for tagseq in fdist.keys():
    print tagseq, fdist[tagseq]

def ch07_13c_better_chunker():
  # can be improved with more patterns from the top from previous method
  from nltk.corpus import conll2000
  grammar = r"""
  NP : {<DT> <JJ> <NN.*>}
       {<DT> <NN.*>}
       {<JJ> <NN.*>}
       {<NN.*>+}
  """
  cp = nltk.RegexpParser(grammar)
  test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
  print cp.evaluate(test_sents)

def _chunk2brackets(sent):
  bracks = []
  for wpi in nltk.chunk.tree2conllstr(sent).split("\n"):
    (word, pos, iob) = wpi.split(" ")
    bracks.append((word, pos))
  return bracks

def _chunk2iob(sent):
  iobs = []
  for wpi in nltk.chunk.tree2conllstr(sent).split("\n"):
    (word, pos, iob) = wpi.split(" ")
    iobs.append((word, pos, iob))
  return iobs

def ch07_16a_penn_treebank():
  from nltk.corpus import treebank_chunk
  for sent in treebank_chunk.chunked_sents("wsj_0001.pos"):
    print "sent=", sent
    print "chunk2brackets=", _chunk2brackets(sent)
    print "chunk2iob=", _chunk2iob(sent)
    
def main():
#  ch07_02_match_np_containing_plural_nouns()
#  ch07_03_develop_grammar_with_chunkparser()
#  ch07_05_tag_pattern_np_containing_gerund()
#  ch07_06_coordinated_noun_phrases()
#  ch07_07_chunker_eval()
#  ch07_13a_tag_seqs_for_np()
#  ch07_13c_better_chunker()
  ch07_16a_penn_treebank()

if __name__ == "__main__":
  main()
