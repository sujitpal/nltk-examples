import nltk
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize
import re

def train_pos_tagger():
  """
  Trains a POS tagger with sentences from Penn Treebank
  and returns it.
  """
  train_sents = treebank.tagged_sents(simplify_tags=True)
  tagger = nltk.TrigramTagger(train_sents, backoff=
    nltk.BigramTagger(train_sents, backoff=
    nltk.UnigramTagger(train_sents, backoff=
    nltk.DefaultTagger("NN"))))
  return tagger

def ce_phrases():
  """
  Returns a list of phrases found using bootstrap.py ordered
  by number of words descending (so code traversing the list
  will encounter the longest phrases first).
  """
  def by_phrase_len(x, y):
    lx = len(word_tokenize(x))
    ly = len(word_tokenize(y))
    if lx == ly:
      return 0
    elif lx < ly:
      return 1
    else:
      return -1
  ceps = []
  phrasefile = open("ce_phrases.txt", 'rb')
  for cep in phrasefile:
    ceps.append(cep[:-1])
  phrasefile.close()
  return map(lambda phrase: word_tokenize(phrase),
    sorted(ceps, cmp=by_phrase_len))

def ce_phrase_words(ce_phrases):
  """
  Returns a set of words in the ce_phrase list. This is
  used to tag words that refer to the NE but does not
  have a consistent pattern to match against.
  """
  ce_words = set()
  for ce_phrase_tokens in ce_phrases:
    for ce_word in ce_phrase_tokens:
      ce_words.add(ce_word)
  return ce_words

def slice_matches(a1, a2):
  """
  Returns True if the two arrays are content wise identical,
  False otherwise.
  """
  if len(a1) != len(a2):
    return False
  else:
    for i in range(0, len(a1)):
      if a1[i] != a2[i]:
        return False
    return True
  
def slots_available(matched_slots, start, end):
  """
  Returns True if all the slots in the matched_slots array slice
  [start:end] are False, ie, available, else returns False.
  """
  return len(filter(lambda slot: slot, matched_slots[start:end])) == 0

def promote_coreferences(tuple, ce_words):
  """
  Sets the io_tag to True if it is not set and if the word is
  in the set ce_words. Returns the updated tuple (word, pos, iotag)
  """
  return (tuple[0], tuple[1],
    True if tuple[2] == False and tuple[0] in ce_words else tuple[2])

def tag(sentence, pos_tagger, ce_phrases, ce_words):
  """
  Tokenizes the input sentence into words, computes the part of
  speech and the IO tag (for whether this word is "in" a CE named
  entity or not), and returns a list of (word, pos_tag, io_tag)
  tuples.
  """
  tokens = word_tokenize(sentence)
  # add POS tags using our trained POS Tagger
  pos_tagged = pos_tagger.tag(tokens)
  # add the IO(not B) tags from the phrases we discovered
  # during bootstrap.
  words = [w for (w, p) in pos_tagged]
  pos_tags = [p for (w, p) in pos_tagged]
  io_tags = map(lambda word: False, words)
  for ce_phrase in ce_phrases:
    start = 0
    while start < len(words):
      end = start + len(ce_phrase)
      if slots_available(io_tags, start, end) and \
          slice_matches(words[start:end], ce_phrase):
        for j in range(start, end):
          io_tags[j] = True
        start = end + 1
      else:
        start = start + 1
  # zip the three lists together
  pos_io_tagged = map(lambda ((word, pos_tag), io_tag):
    (word, pos_tag, io_tag), zip(zip(words, pos_tags), io_tags))
  # "coreference" handling. If a single word is found which is
  # contained in the set of words created by our phrases, set
  # the IO(not B) tag to True if it is False
  return map(lambda tuple: promote_coreferences(tuple, ce_words),
    pos_io_tagged)

shape_A = re.compile("[A-Zbdfhklt0-9#$&/@|]")
shape_x = re.compile("[acemnorsuvwxz]")
shape_i = re.compile("[i]")
shape_g = re.compile("[gpqy]")
shape_j = re.compile("[j]")

def shape(word):
  wbuf = []
  for c in word:
    wbuf.append("A" if re.match(shape_A, c) != None
      else "x" if re.match(shape_x, c) != None
      else "i" if re.match(shape_i, c) != None
      else "g" if re.match(shape_g, c) != None
      else "j")
  return "".join(wbuf)

def word_features(tagged_sent, wordpos):
  return {
    "word": tagged_sent[wordpos][0],
    "pos": tagged_sent[wordpos][1],
    "prevword": "<START>" if wordpos == 0 else tagged_sent[wordpos-1][0],
    "prevpos": "<START>" if wordpos == 0 else tagged_sent[wordpos-1][1],
    "nextword": "<END>" if wordpos == len(tagged_sent)-1
                        else tagged_sent[wordpos+1][0],
    "nextpos": "<END>" if wordpos == len(tagged_sent)-1
                       else tagged_sent[wordpos+1][1],
    "shape": shape(tagged_sent[wordpos][0])
  }
