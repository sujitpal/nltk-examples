#!/usr/bin/python

from __future__ import division
from operator import itemgetter
import nltk

def ch05_1_3_tag_sentences():
  sents = [
    "British left waffles on Falkland Islands.",
    "Juvenile Court to try shooting defendant.",
    "They wind back the clock, while we chase after the wind."
  ]
  for sent in sents:
    tokens = nltk.word_tokenize(sent)
    print nltk.pos_tag(tokens)

def ch05_10_train_test_unigram_tagger():
  from nltk.corpus import brown
  fd = nltk.FreqDist(brown.words(categories="news"))
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
  most_freq_pos = dict((word, cfd[word].max()) for word in fd.keys())
  unigram_tagger = nltk.UnigramTagger(model=most_freq_pos)
  for sent in brown.sents(categories="editorial")[:10]:
    tagged = unigram_tagger.tag(sent)
    print sent
    print ">>>", tagged
    print "not tagged: ", filter(lambda (a,b): b == None, tagged)

def ch05_11_train_test_affix_tagger():
  from nltk.corpus import brown
  fd = nltk.FreqDist(brown.words(categories="news"))
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
  most_freq_pos = dict((word, cfd[word].max()) for word in fd.keys())
  affix_tagger = nltk.AffixTagger(model=most_freq_pos)
  print affix_tagger.evaluate(brown.tagged_sents(categories="editorial"))

def ch05_14_brown_corpus_tags_list():
  from nltk.corpus import brown
  print sorted(set([t for (w, t) in brown.tagged_words()]))

def ch05_15_brown_corpus_trivia():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  # which nouns are more common in plural form than singular?
  # NNS - plural, NN - singular. Calculate plural = singular + s
  s_nouns = [w for (w,t) in tagged_words if t == "NN"]
  plurals = set([w + "s" for w in s_nouns])
  p_nouns = [w for (w,t) in tagged_words if t == "NNS" and w in plurals]
  s_fd = nltk.FreqDist(s_nouns)
  p_fd = nltk.FreqDist(p_nouns)
  print "words where singular > plural=", \
    filter(lambda word: s_fd[word] < p_fd[word], p_fd.keys())[:50]
  # which word has the greatest number of distinct tags
  word_tags = nltk.defaultdict(lambda: set())
  for word, token in tagged_words:
    word_tags[word].add(token)
  ambig_words = sorted([(k, len(v)) for (k, v) in word_tags.items()],
    key=itemgetter(1), reverse=True)[:50]
  print [(word, numtoks, word_tags[word]) for (word, numtoks) in ambig_words]
  # list top 20 (by frequency) tags
  token_fd = nltk.FreqDist([token for (word, token) in tagged_words])
  print "top_tokens=", token_fd.keys()[:20]
  # which tags are nouns most commonly found after
  tagged_word_bigrams = nltk.bigrams(tagged_words)
  fd_an = nltk.FreqDist([t1 for (w1,t1),(w2,t2)
    in tagged_word_bigrams if t2.startswith("NN")])
  print "nouns commonly found after these tags:", fd_an.keys()

def ch05_17_lookup_tagger_performance_upper_limit():
  # average percentage of words that are assigned the most likely
  # tokens for the word
  from nltk.corpus import brown
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
  sum_of_avgs = 0
  for word in cfd.conditions():
    mlt = reduce(lambda t1, t2: t1 if t1 > t2 else t2, cfd[word])
    num_mlt_tags = cfd[word][mlt]
    num_all_tags = cfd[word].N()
    sum_of_avgs += num_mlt_tags / num_all_tags
  print "perf_upper_limit=", sum_of_avgs / len(cfd.conditions())

def ch05_18_brown_corpus_statistics():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  vocab_size = len(set([w for (w,t) in tagged_words]))
  cfd = nltk.ConditionalFreqDist(tagged_words)
  # proportion of word types always assigned the same part-of-speech
  # ie words with a single POS
  num_single_pos_words = sum(len(cfd[word].hapaxes())
    for word in cfd.conditions())
  print "prop of word types with single POS=", \
    num_single_pos_words / vocab_size
  # how many words are ambiguous, ie with >= 2 POS tags
  ambig_words = [w for w in cfd.conditions()
    if len(filter(lambda x: cfd[w][x] >= 2, cfd[w].keys())) >= 2]
  num_ambig_words = len(ambig_words)
  print "prop of ambiguous words (>= 2 POS)=", \
    num_ambig_words / vocab_size
  # percentage of word tokens in the brown corpus that involve
  # ambiguous words
  token_size = len(set([t for (w,t) in tagged_words]))
  unique_tokens = set()
  for w in ambig_words:
    unique_tokens.update(set([t for t in cfd[w].keys()]))
  print "prop of ambig tokens=", len(unique_tokens) / token_size

def ch05_20_brown_corpus_words_phrases_by_tag():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  # produce alpha sorted list of distinct words tagged MD
  print sorted(set([w.lower()
    for (w,t) in filter(lambda (w,t): t == "MD", tagged_words)]))
  # identify words that can be plural (NRS, NPS*, NNS*) or
  # third person singular verbs (BEDZ*, BEZ*, DOZ*, *BEZ)
  # AND the ones ending with "s"
  print set([w for (w, t) in tagged_words
    if w.lower().endswith("s") and
    (t == "NRS" or t.startswith("NPS")
    or t.startswith("NPS") or t.startswith("NNS")
    or t.startswith("BEDZ") or t.startswith("BEZ")
    or t.startswith("DOZ") or t.endswith("BEZ"))])
  # identify 3 word prepositional phrases IN+DET+NN
  tagged_word_trigrams = nltk.trigrams(tagged_words)
  print tagged_word_trigrams[:10]
  print set([" ".join([w1, w2, w3])
    for (w1,t1), (w2,t2), (w3,t3) in tagged_word_trigrams
    if t1 == "IN" and t2 == "DET" and t3 == "NN"])
  # ratio of masculine to feminine pronouns
  num_masc_pn = len([w for (w,t) in tagged_words if w.lower() == "he"])
  num_fem_pn = len([w for (w,t) in tagged_words if w.lower() == "she"])
  print "masc/fem = ", (num_masc_pn / num_fem_pn)

def ch05_21_qualifiers_before_adore_love_like_prefer():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  tagged_word_bigrams = nltk.bigrams(tagged_words)
  allp = set(["adore", "love", "like", "prefer"])
  print set([w for (w1,t1), (w2,t2) in tagged_word_bigrams
    if t1 == "QL" and w2.lower() in allp])

def ch05_22_regular_expression_tagger():
  from nltk.corpus import brown
  tagged_sents = brown.tagged_sents(categories="news")
  patterns = [ # patterns copied from page 199
    (r".*s$", "NNS"), # plurals
    (r".*ing$", "VBG"), # gerund
    (r".*ould$", "MD"), # modal
    (r".*ed$", "VBD"), # verb past
    (r".*es$", "VBZ"), # 3rd person singular
    (r'.*', "NN")       # fallback to noun
  ]
  tagger = nltk.RegexpTagger(patterns)
  print tagger.evaluate(tagged_sents)

def ch05_27_collapse_tags_based_on_conf_matrix():
  # TODO: run ch05.py:ambiguous_tags to get confusion matrix
  print "TODO"
  
def ch05_30_bigram_tagger_low_freq_words_as_unk():
  from nltk.corpus import brown
  # before UNK, check tagger score
  sents = brown.tagged_sents(categories="news")
  size = int(len(sents) * 0.9)
  train_sents = sents[:size]
  test_sents = sents[size:]
  tagger1 = nltk.BigramTagger(train_sents)
  print "before UNK, evaluate=", tagger1.evaluate(test_sents)
  # replace low freq words with UNK
  words = brown.tagged_words(categories="news")
  fd = nltk.FreqDist([w for (w,t) in words])
  lfw = set([w for (w,t) in words if fd[w] <= 1])
  sents2 = []
  for sent in train_sents:
    sents2.append(map(lambda (w,t): ("UNK",t) if w in lfw else (w,t), sent))
  tagger2 = nltk.BigramTagger(sents2)
  print "after UNK, evaluate=", tagger2.evaluate(test_sents)

def ch05_32_brill_tagger():
  # TODO: check out usage of brill tagger
  # also see # 40
  print "TODO"

def ch05_33_list_pos_of_word_given_word_and_pos():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  tagged_word_bigrams = nltk.bigrams(tagged_words)
  dd = nltk.defaultdict(dict)
  for (w1,t1), (w2,t2) in tagged_word_bigrams:
    dd[w1][t1] = t2
  print dd

def ch05_34_num_words_with_1to10_distinct_tags():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  # number of distinct tags and number of words in corpus for this
  dd = nltk.defaultdict(set)
  for w,t in tagged_words:
    dd[w].add(t)
  for i in range(1,10):
    print i, len(filter(lambda x: len(dd[x]) == i, dd.keys()))
  # for the word with greatest number of tags, print out concordance
  # one for each tag
  maxtags = 6
  word = None
  tags = None
  for w in dd.keys():
    if len(dd[w]) >= maxtags:
      word = w
      tags = dd[w]
      break
  poss = []
  pos = 0
  for w, t in tagged_words:
    if w == word and t in tags:
      poss.append((t, pos))
      tags.remove(t)
    pos += 1
  for t, pos in poss:
    print t, " ".join(w for w,t in tagged_words[pos-10:pos+10])

def ch05_35_must_contexts():
  from nltk.corpus import brown
  tagged_words = brown.tagged_words(categories="news")
  tagged_word_bigrams = nltk.bigrams(tagged_words)
  fd = nltk.FreqDist((w1,t2) for (w1,t1),(w2,t2)
    in tagged_word_bigrams
    if w1 == "must")
  for t in fd.keys():
    print t, fd[t]
  # TODO: epistemic and deontic uses of must?

def ch05_37_prev_token_tagger():
  # TODO
  pass

def ch05_39_statistical_tagger():
  # TODO
  pass

  
def main():
#  ch05_1_3_tag_sentences()
#  ch05_10_train_test_unigram_tagger()
#  ch05_11_train_test_affix_tagger()
#  ch05_14_brown_corpus_tags_list()
#  ch05_15_brown_corpus_trivia()
#  ch05_17_lookup_tagger_performance_upper_limit()
#  ch05_18_brown_corpus_statistics()
#  ch05_20_brown_corpus_words_phrases_by_tag()
#  ch05_21_qualifiers_before_adore_love_like_prefer()
#  ch05_22_regular_expression_tagger()
#  ch05_30_bigram_tagger_low_freq_words_as_unk()
#  ch05_32_brill_tagger()
#  ch05_33_list_pos_of_word_given_word_and_pos()
#  ch05_34_num_words_with_1to10_distinct_tags()
#  ch05_35_must_contexts()
  ch05_36_tagger_training()
  print "end"
  
if __name__ == "__main__":
  main()
