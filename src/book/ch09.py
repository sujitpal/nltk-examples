#!/usr/bin/python
# Building feature based grammars
from __future__ import division
import nltk
import re

def _grammatical_lex2fs(word):
  kim = {"CAT": "NP", "ORTH": "Kim", "REF": "k"}
  chase = {"CAT": "V", "ORTH": "chased", "REL": "chase"}
  lee = {"CAT": "NP", "ORTH": "Lee", "REF": "l"}
  for fs in [kim, lee, chase]:
    if fs["ORTH"] == word:
      return fs

def grammatical_features():
  tokens = "Kim chased Lee".split()
  subj, verb, obj = _grammatical_lex2fs(tokens[0]), \
    _grammatical_lex2fs(tokens[1]), _grammatical_lex2fs(tokens[2])
  verb["AGT"] = subj["REF"] # agent of chase is Kim
  verb["PAT"] = obj["REF"]  # patient of chase is Lee
  for k in ["ORTH", "REL", "AGT", "PAT"]:
    print "%-5s => %s" % (k, verb[k])

def the_dog_runs():
  grammar1 = """
    S -> NP VP
    NP -> Det N
    VP - V
    Det -> 'this'
    N -> 'dog'
    V -> 'runs'
  """
  grammar2 = """
    S -> NP_SG VP_SG
    S -> NP_PL VP_PL
    NP_SG -> Det_SG N_SG
    NP_PL -> Det_PL N_PL
    VP_SG -> V_SG
    VP_PL -> V_PL
    Det_SG -> 'this'
    Det_PL -> 'these'
    N_SG -> 'dog
    N_PL -> 'dogs'
    V_SG -> 'runs'
    V_PL -> 'run'
  """
  grammar3 = """
    S -> NP[NUM=?n] VP[NUM=?n]
    S -> NP_PL VP_PL
    NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
    NP_PL -> Det_PL N_PL
    VP[NUM=?n] -> V[NUM=?n]
    VP_PL -> V_PL
    Det_SG -> 'this'
    Det_PL -> 'these'
    N_SG -> 'dog
    N_PL -> 'dogs'
    V_SG -> 'runs'
    V_PL -> 'run'
  """

def sample_grammar():
  nltk.data.show_cfg("grammars/book_grammars/feat0.fcfg")
  tokens = "Kim likes children".split()
  from nltk import load_parser
  cp = load_parser("grammars/book_grammars/feat0.fcfg", trace=2)
  trees = cp.nbest_parse(tokens)

def feature_structures():
  fs1 = nltk.FeatStruct(TENSE='past', NUM='sg')
  print "fs1=", fs1
  print "fs1[TENSE]=", fs1['TENSE']
  fs1['CASE'] = 'acc'
  fs2 = nltk.FeatStruct(POS='N', AGR=fs1)
  print "fs2=", fs2
  person = nltk.FeatStruct(name='Lee', telno='212 444 1212', age=33)
  print "person=", person
  print nltk.FeatStruct("""
  [NAME='Lee', ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
  SPOUSE=[Name='Kim', ADDRESS->(1)]]
  """)

def feature_structure_unification():
  fs1 = nltk.FeatStruct(NUMBER=74, STREE='rue Pascal')
  fs2 = nltk.FeatStruct(CITY='Paris')
  print fs1.unify(fs2)
  # result of unification if fs1 subsumes fs2 or vice versa, the more
  # specific of the two.
  fs0 = nltk.FeatStruct("""
    [NAME='Lee', ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
    SPOUSE=[Name='Kim', ADDRESS->(1)]]
  """)
  print "fs0=", fs0
  fs1 = nltk.FeatStruct("[SPOUSE=[ADDRESS=[CITY=Paris]]]")
  print fs1.unify(fs0)
  print "fs1=", fs1
  fs2 = nltk.FeatStruct("""
    [NAME=Lee, ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
    SPOUSE=[NAME=Kim, ADRRESS->(1)]]
  """)
  print "fs1.unify(fs2)=", fs1.unify(fs2)
  fs3 = nltk.FeatStruct("[ADDRESS=?x, ADDRESS2=?x]")
  print "fs2.unify(fs3)=", fs2.unify(fs3)

def sentence_parsing():
#  tokens = "who do you claim that you like".split()
#  tokens = "you claim that you like cats".split()
  tokens = "rarely do you sing".split()
  from nltk import load_parser
  cp = load_parser("grammars/book_grammars/feat1.fcfg")
  for tree in cp.nbest_parse(tokens):
    print tree
    tree.draw()
    
def main():
#  grammatical_features()
#  the_dog_runs()
#  sample_grammar()
#  feature_structures()
#  feature_structure_unification()
  sentence_parsing()
  print "end"
  
if __name__ == "__main__":
  main()
