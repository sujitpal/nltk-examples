#!/usr/bin/python
# Analyzing sentence structure

from __future__ import division
import nltk
import re

def sentence_parse_example():
  groucho_grammar = nltk.parse_cfg("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
  """)
  sent = ["I", "shot", "an", "elephant", "in", "my", "pajamas"]
  parser = nltk.ChartParser(groucho_grammar)
  trees = parser.nbest_parse(sent)
  for tree in trees:
    print tree

def simple_cfg():
#  grammar = nltk.parse_cfg("""
#    S -> NP VP
#    VP -> V NP | V NP PP
#    PP -> P NP
#    V -> "saw" | "ate" | "walked"
#    NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
#    Det -> "a" | "an" | "the" | "my"
#    N -> "man" | "dog" | "cat" | "telescope" | "park"
#    P -> "in" | "on" | "by" | "with"
#  """)
  # also can load grammar from text file
  # grammar = nltk.data.load("file:mygrammar.cfg")
  grammar = nltk.parse_cfg("""
    S -> NP VP
    NP -> Det Nom | PropN
    Nom -> Adj Nom | N
    VP -> V Adj | V NP | V S | V NP PP
    PP -> P NP
    PropN -> 'Buster' | 'Chatterer' | 'Joe'
    Det -> 'the' | 'a'
    N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
    Adj -> 'angry' | 'frightened' | 'little' | 'tall'
    V -> 'chased' | 'saw' | 'said' | 'thought' | 'was' | 'put'
    P -> 'on'
  """)
#  sent = "Mary saw Bob".split()
  # structural ambiguity - 2 parse trees for this.
  # prepositional phrase attach ambiguity.
#  sent = "the dog saw a man in a park".split()
  # For second grammar
#  sent = "the angry bear chased the frightened little squirrel".split()
  sent = "Chatterer said Buster thought the tree was tall".split()
#  rd_parser = nltk.RecursiveDescentParser(grammar, trace=2) # for debug
  # NOTE: production rules need to be right-recursive, ie X -> Y X
  rd_parser = nltk.RecursiveDescentParser(grammar)
  for tree in rd_parser.nbest_parse(sent):
    print tree

# recursive descent parsing - top down
#   nltk.app.rdparser() - recursive descent demo
#   shortcomings - left recursive productions result in infinite loop
#                  parser wastes time considering paths that it discards
#                  backtracking discards what may need to be rebuilt
# shift-reduce - bottom up
#    nltk.app.srparser() - demo
#    can reach dead end and fail to find a parse
#    with Lookahead LR parser
#    only builds structure corresponding to word in input.
# left-corner filtering - top down w/ bottom up filtering
#    each production is stored along with its left corner element on RHS
#    eg, S -> NP VP; VP -> V NP | ... => (S,NP), (VP,V), ...
# chart parsing - dynamic programming
#    nltk.app.chartparser()
def parsing_types():
  grammar = nltk.parse_cfg("""
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP
    V -> "saw" | "ate" | "walked"
    NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
    Det -> "a" | "an" | "the" | "my"
    N -> "man" | "dog" | "cat" | "telescope" | "park"
    P -> "in" | "on" | "by" | "with"
  """)
  sent = "Mary saw a dog".split()
  rd_parser = nltk.RecursiveDescentParser(grammar)
  print "==== recursive descent ===="
  for t in rd_parser.nbest_parse(sent):
    print t
  sr_parser = nltk.ShiftReduceParser(grammar)
  print "==== shift reduce ===="
  for t in sr_parser.nbest_parse(sent):
    print t

def _chart_init_wfst(tokens, grammar):
  numtokens = len(tokens)
  wfst = [[None for i in range(numtokens+1)] for j in range(numtokens+1)]
  for i in range(numtokens):
    productions = grammar.productions(rhs = tokens[i])
    wfst[i][i+1] = productions[0].lhs()
  return wfst

def _chart_complete_wfst(wfst, tokens, grammar, trace=False):
  index = dict((p.rhs(), p.lhs()) for p in grammar.productions())
  numtokens = len(tokens)
  for span in range(2, numtokens+1):
    for start in range(numtokens+1-+span):
      end = start + span
      for mid in range(start+1, end):
        nt1, nt2 = wfst[start][mid], wfst[mid][end]
        if nt1 and nt2 and (nt1,nt2) in index:
          wfst[start][end] = index[(nt1, nt2)]
          if trace:
            print "[%s] %3s [%s] %3s [%s] ==> [%s] %3s [%s]" % \
              (start, nt1, mid, nt2, end, start, index[(nt1,nt2)], end)
  return wfst

def _chart_display(wfst, tokens):
  print "\nWFST " + " ".join([("%-4d" %i) for i in range(1, len(wfst))])
  for i in range(len(wfst)-1):
    print "%-4d" % i,
    for j in range(1, len(wfst)):
      print "%-4s" % (wfst[i][j] or "."),
    print
    
def chart_parsing():
  groucho_grammar = nltk.parse_cfg("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
  """)
  tokens = "I shot an elephant in my pajamas".split()
  wfst0 = _chart_init_wfst(tokens, groucho_grammar)
  _chart_display(wfst0, tokens)
  wfst1 = _chart_complete_wfst(wfst0, tokens, groucho_grammar, trace=True)
  _chart_display(wfst1, tokens)

def dependency_grammar():
  groucho_dep_grammar = nltk.parse_dependency_grammar("""
    'shot' -> 'I' | 'elephant' | 'in'
    'elephant' -> 'an' | 'in'
    'in' -> 'pajamas'
    'pajamas' -> 'my'
  """)
  print groucho_dep_grammar
  pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
  sent = "I shot an elephant in my pajamas".split()
  trees = pdp.parse(sent)
  for tree in trees:
#    tree.draw()
    print tree

def _grammar_filter(tree):
  child_nodes = [child.node for child in tree
    if isinstance(child, nltk.Tree)]
  return (tree.node == "VP") and ("S" in child_nodes)

def grammar_development_with_treebank():
  from nltk.corpus import treebank
  t = treebank.parsed_sents("wsj_0001.mrg")[0]
  print t
  print "identify verbs for SV in VP -> SV S", [subtree for tree
    in treebank.parsed_sents()
    for subtree in tree.subtrees(_grammar_filter)]

def word_valency():
  table = nltk.defaultdict(lambda: nltk.defaultdict(set))
  entries = nltk.corpus.ppattach.attachments("training")
  for entry in entries:
#    print entry
    key = entry.noun1 + "-" + entry.prep + "-" + entry.noun2
    table[key][entry.attachment].add(entry.verb)
  for key in sorted(table):
    if len(table[key]) > 1:
      print key, "N:", sorted(table[key]["N"]), "V:", sorted(table[key]["V"])

def _give_give(t):
  return t.node == "VP" and len(t) > 3 and t[1].node == "NP" and \
    (t[2].node == "PP-DIV" or t[2].node == "NP") and \
    ("give" in t[0].leaves() or "gave" in t[0].leaves())

def _give_sent(t):
  return " ".join(token for token in t.leaves() if token[0] not in "*-O")

def _give_print_node(t, width):
  output = "%s %s: %s / %s: %s" % \
    (_give_sent(t[0]), t[1].node, _give_sent(t[1]), t[2].node, _give_sent(t[2]))
  if len(output) > width:
    output = output[:width] + "..."
  print output

def give_gave_usage():
  # Kim gave a bone to the dog VS Kim gave the dog a bone (equally likely)
  # Kim gives the heebie-jeebies to me LESS LIKELY THAN
  # Kim gives me the heebie-jeebies.
  for tree in nltk.corpus.treebank.parsed_sents():
    for t in tree.subtrees(_give_give):
      _give_print_node(t, 72)

def pcfg_parser():
#  grammar = nltk.parse_pcfg("""
#    S -> NP VP         [1.0]
#    VP -> TV NP        [0.4]
#    VP -> IV           [0.3]
#    VP -> DatV NP NP   [0.3]
#    TV -> 'saw'        [1.0]
#    IV -> 'ate'        [1.0]
#    DatV -> 'gave'     [1.0]
#    NP -> 'telescopes' [0.8]
#    NP -> 'Jack'       [0.2]
#  """)
  # alternative repr, or clause probs must sum to 1
  grammar = nltk.parse_pcfg("""
    S -> NP VP         [1.0]
    VP -> TV NP [0.4] | IV [0.3] | DatV NP NP [0.3]
    TV -> 'saw'        [1.0]
    IV -> 'ate'        [1.0]
    DatV -> 'gave'     [1.0]
    NP -> 'telescopes' [0.8]
    NP -> 'Jack'       [0.2]
  """)
  print grammar
  viterbi_parser = nltk.ViterbiParser(grammar)
  print viterbi_parser.parse("Jack saw telescopes".split())
  
def main():
#  sentence_parse_example()
#  simple_cfg()
#  parsing_types()
#  chart_parsing()
#  dependency_grammar()
#  grammar_development_with_treebank()
#  word_valency()
#  give_gave_usage()
  pcfg_parser()
  print "end"
  
if __name__ == "__main__":
  main()
