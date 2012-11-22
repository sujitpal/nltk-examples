#!/usr/bin/python
# Extracting information from text

from __future__ import division
import nltk
import re

def _ie_preprocess(document):
  sentences = nltk.sent_tokenize(document)
  sentences = [nltk.word_tokenizer(sent) for sent in sentences]
  sentences = [nltk.os_tag(sent) for sent in sentences]

def simple_regex_based_np_chunker():
  sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
    ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"),
    ("cat", "NN")]
  grammar = r"""
    NP : {<DT|PP\$>?<JJ>*<NN>}   # determiner/possessive, adjective, noun
         {<NNP>+}                # sequences of proper nouns
         {<NN>+}                 # sequence of common nouns
         }<VBD|IN>+{             # Chink sequences of VBD and IN
  """
  cp = nltk.RegexpParser(grammar)
  result = cp.parse(sentence)
  print result
#  result.draw()
  nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
  print cp.parse(nouns)
  sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
      ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
  print cp.parse(sentence)

def _find_chunks(pattern):
  print "======", pattern, "======="
  cp = nltk.RegexpParser(r"""
    CHUNK: {%s}
  """ % (pattern))
  brown = nltk.corpus.brown
  for sent in brown.tagged_sents():
    tree = cp.parse(sent)
    for subtree in tree.subtrees():
      if subtree.node == "CHUNK":
        print subtree

def extract_pos_pattern_with_chunk_parser():
  _find_chunks("<V.*> <TO> <V.*>")
  _find_chunks("<N.*> <N.*> <N.*> <N.*>+")

def iob_to_tree():
  text = """
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
  """
  tree = nltk.chunk.conllstr2tree(text, chunk_types=["NP"])
  print tree

def read_chunked_corpus():
  from nltk.corpus import conll2000
  print conll2000.chunked_sents("train.txt")[99]
  print conll2000.chunked_sents("train.txt", chunk_types = ["NP"])[99]

def evaluate_chunker():
  from nltk.corpus import conll2000
  cp = nltk.RegexpParser("") # baseline
  test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
  print cp.evaluate(test_sents)
  grammar = r"NP: {<[CDJNP].*>+}"
  cp1 = nltk.RegexpParser(grammar) # naive tagger, look for all tags in NP chunk
  print cp1.evaluate(test_sents)

class UnigramChunker(nltk.ChunkParserI):
  def __init__(self, train_sents):
    train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                    for sent in train_sents]
    self.tagger = nltk.UnigramTagger(train_data)
#    self.tagger = nltk.BigramTagger(train_data) # increase accuracy a bit

  def parse(self, sentence):
    pos_tags = [pos for (word,pos) in sentence]
    tagged_pos_tags = self.tagger.tag(pos_tags)
    chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
    conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                  in zip(sentence, chunktags)]
    return nltk.chunk.conlltags2tree(conlltags)

def chunk_with_unigram_tagger():
  # use unigram tagger to find the IOB tag given its POS tag
  from nltk.corpus import conll2000
  test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
  train_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
  unigram_chunker = UnigramChunker(train_sents)
  print unigram_chunker.evaluate(test_sents)
  postags = sorted(set(pos for sent in train_sents
                           for (word, pos) in sent.leaves()))
  print unigram_chunker.tagger.tag(postags)

def _npchunk_features(sentence, i, history):
  features = {}
  word, pos = sentence[i]
  features["pos"] = pos
  # add previous POS tag
  prevword, prevpos = "<START>", "<START>" if i == 0 else sentence[i-1]
  features["prevpos"] = prevpos
  # add current word
  features["word"] = word
  # more features
  nextword, nextpos = "<END>", "<END>" if i == len(sentence) - 1 else sentence[i+1]
  features["nextpos"] = nextpos
  features["prevpos+pos"] = "%s+%s" % (prevpos, pos)
  features["pos+nextpos"] = "%s+%s" % (pos, nextpos)
  # tags since last determiner
  tags_since_dt = set()
  for word, pos in sentence[:i]:
    if pos == "DT":
      tags_since_dt = set()
    else:
      tags_since_dt.add(pos)
  features["tags_since_dt"] = "+".join(sorted(tags_since_dt))
  return features

class ConsecutiveNPChunkTagger(nltk.TaggerI):
  def __init__(self, train_sents):
    train_set = []
    for tagged_sent in train_sents:
      untagged_sent = nltk.tag.untag(tagged_sent)
      history = []
      for i, (word, tag) in enumerate(tagged_sent):
        featureset = _npchunk_features(untagged_sent, i, history)
        train_set.append((featureset, tag))
        history.append(tag)
    self.classifier = nltk.MaxentClassifier.train(train_set,
      algorithm="GIS", trace=0)

  def tag(self, sentence):
    history = []
    for i, word in enumerate(sentence):
      featureset = _npchunk_features(sentence, i, history)
      tag = self.classifier.classify(featureset)
      history.append(tag)
    return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
  def __init__(self, train_sents):
    tagged_sents = [[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
    self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

  def parse(self, sentence):
    tagged_sents = self.tagger.tag(sentence)
    conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
    return nltk.chunk.conlltags2tree(conlltags)

def train_classifier_based_chunker():
  from nltk.corpus import conll2000
  test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
  train_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
  chunker = ConsecutiveNPChunker(train_sents)
  print chunker.evaluate(test_sents)

def recursive_chunk_parser():
  grammar = r"""
    NP : {<DT|JJ|NN.*>+}    # chunk sentences of DT,JJ,NN
    PP : {<IN><NP>}         # chunk preposition followed by NP
    VP : {<VB.*><NP|PP|CLAUSE>+$}  # chunk verb and their argument
    CLAUSE : {<NP><VP>}     # chunk NP,VP
  """
  cp = nltk.RegexpParser(grammar, loop=2) # parses sentence multiple times
  sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
    ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
  print cp.parse(sentence)
  sentence2 = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NNP"),
    ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
    ("on", "IN"), ("the", "DT"), ("mat", "NN")]
  print cp.parse(sentence2)

def _traverse(t):
  try:
    t.node
  except AttributeError:
    print t,
  else:
    print "(", t.node,
    for child in t:
      _traverse(child)
    print ")",
    
def nltk_tree_handling():
  # construction
  tree1 = nltk.Tree("NP", ["Alice"])
  print "tree1=", tree1
  tree2 = nltk.Tree("NP", ["the", "rabbit"])
  print "tree2=", tree2
  tree3 = nltk.Tree("VP", ["chased", tree2])
  print "tree3=", tree3
  tree4 = nltk.Tree("S", [tree1, tree3])
  print "tree4=", tree4
  # deconstruction
  print "tree4[1]=", tree4[1]
  print "tree4[1].node=", tree4[1].node, \
    "tree4[1].leaves()=", tree4[1].leaves()
  print "tree4[1][1][1]=", tree4[1][1][1]
  _traverse(tree4)

def named_entity_recognition():
  # Gazetteers: Alexandria or Getty
  sent = nltk.corpus.treebank.tagged_sents()[22]
  print "NE (binary=True)", nltk.ne_chunk(sent, binary=True)
  print "indiv NE types (binary=False)", nltk.ne_chunk(sent)

def relation_extraction():
  IN = re.compile(r".*\bin\b(?!\b.+ing)")
  for doc in nltk.corpus.ieer.parsed_docs("NYT_19980315"):
    for rel in nltk.sem.extract_rels("ORG", "LOC", doc, corpus="ieer", pattern=IN):
      print nltk.sem.show_raw_rtuple(rel)

def relation_extraction2():
  # needs POS as well as NE annotations (in Dutch)
  from nltk.corpus import conll2002
  vnv = """
(
is/V|       # 3rd sing present and
was/V|      # past forms of the verm zijn (be)
werd/V|     # and also present
wordt/V     # past of worden (become)
).*           # followed by anything
van/Prep      # followed by van (of)
  """
  VAN = re.compile(vnv, re.VERBOSE)
  for doc in conll2002.chunked_sents("ned.train"):
    for r in nltk.sem.extract_rels("PER", "ORG", doc,
        corpus="conll2002", pattern=VAN):
#      print nltk.sem.show_clause(r, relsym="VAN")
      print nltk.sem.show_raw_rtuple(r, lcon=True, rcon=True)

def main():
  simple_regex_based_np_chunker()
#  extract_pos_pattern_with_chunk_parser()
#  iob_to_tree()
#  read_chunked_corpus()
#  evaluate_chunker()
#  chunk_with_unigram_tagger()
#  train_classifier_based_chunker() # TODO: finish running
#  recursive_chunk_parser()
#  nltk_tree_handling()
#  named_entity_recognition()
#  relation_extraction()
#  relation_extraction2()
  print "end"
  
if __name__ == "__main__":
  main()
