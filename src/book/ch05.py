#!/usr/bin/python
# Categorizing and tagging words

from __future__ import division
import nltk

def basic_tagger_usage():
  texts = [
    "And now for something completely different.",
    "They refuse to permit us to obtain the refuse permit.",
    "I went to the bathroom to flush the toilet.",
    "His face was flushed with fever."
  ]
  for text in texts:
    tokens = nltk.word_tokenize(text)
    print text
    print ">>>", nltk.pos_tag(tokens)

# finds similar POS words based on context
def similar_words():
  text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
  print "similar(woman)=", text.similar("woman")
  print "similar(bought)=", text.similar("bought")
  print "similar(over)=", text.similar("over")

def tagged_token_representation():
  print nltk.tag.str2tuple("fly/NN")
  from nltk.corpus import brown
  print brown.tagged_words()
  # distribution of tags
  brown_news_tagged = brown.tagged_words(categories="news", simplify_tags=True)
  tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
  print tag_fd
  tag_fd.plot(cumulative=True)
  # distribution of POS+N pairs
  word_tag_pairs = nltk.bigrams(brown_news_tagged)
  print nltk.FreqDist(a[1] for (a, b) in word_tag_pairs if b[1] == "N")

def common_verbs_in_news():
  wsj = nltk.corpus.treebank.tagged_words(simplify_tags=True)
  word_tag_fd = nltk.FreqDist(wsj)
  print [word + "/" + tag for (word, tag) in word_tag_fd if tag.startswith("N")]
  cfd1 = nltk.ConditionalFreqDist(wsj)
  print "cfd1[yield]=", cfd1["yield"].keys()
  print "cfd1[cut]=", cfd1["cut"].keys()
  cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
  print "cfd2[VN]=", cfd2["VN"].keys() # past participle
  # find words which can be either past tense (VD) and past participle (VN)
  print [w for w in cfd1.conditions() if "VD" in cfd1[w] and "VN" in cfd1[w]]
  idx1 = wsj.index(("kicked", "VD"))
  print "context(kicked/VD)=", wsj[idx1-4:idx1+1]
  idx2 = wsj.index(("kicked", "VN"))
  print "context(kicked/VN)=", wsj[idx2-4:idx2+1]
  # immediately preceding (word/tag) pairs for cfd2["VN"]
  pfd = nltk.FreqDist(
    wsj[wsj.index((w, "VN")) - 1][0] for w in cfd2["VN"].keys())
  print pfd

def findtags(tag_prefix, tagged_text):
  cfd = nltk.ConditionalFreqDist(
    (tag, word)
    for (word, tag) in tagged_text
    if tag.startswith(tag_prefix))
  return dict(
    (tag, cfd[tag].keys()[:5])
    for tag in cfd.conditions())

def how_is_often_used_in_text():
  from nltk.corpus import brown
  brown_learned_text = brown.words(categories="learned")
  print sorted(set(b for (a, b)
    in nltk.bigrams(brown_learned_text) if a == "often"))
  # or use the tagged words for the actual POS tags
  brown_learned_tagged = brown.tagged_words(categories="learned",
    simplify_tags=True)
  fd = nltk.FreqDist([b[1] for (a,b)
    in nltk.bigrams(brown_learned_tagged) if a[0] == "often"])
  fd.tabulate()

def find_verb_to_verb_patterns():
  from nltk.corpus import brown
  for tagged_sent in brown.tagged_sents():
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(tagged_sent):
      if t1.startswith("V") and t2 == "TO" and t3.startswith("V"):
        print w1, w2, w3

def find_highly_ambiguous_words():
  from nltk.corpus import brown
  brown_news_tagged = brown.tagged_words(
    categories="news", simplify_tags=True)
  cfd = nltk.ConditionalFreqDist(
    (word.lower(), tag)
    for (word, tag) in brown_news_tagged)
  for word in cfd.conditions():
    if len(cfd[word]) > 3:
      tags = cfd[word].keys()
      print word, ":", " ".join(tags)

def tag_most_frequent_words():
  alice = nltk.corpus.gutenberg.words("carroll-alice.txt")
  vocab = nltk.FreqDist(alice)
  v1000 = list(vocab)[:1000]
  mapping = nltk.defaultdict(lambda: "UNK")
  for v in v1000:
    mapping[v] = v
  alice2 = [mapping[v] for v in alice]
  print alice2[:100]

def word_count():
  from nltk.corpus import brown
  counts = nltk.defaultdict(int)
  for (word, tag) in brown.tagged_words(categories="news"):
    counts[tag] +=1
  from operator import itemgetter
  print sorted(counts.items(), key=itemgetter(1), reverse=True)

def anagrams():
  words = nltk.corpus.words.words("en")
#  anagrams = nltk.defaultdict(list)
#  for word in words:
#    key = "".join(sorted(word))
#    anagrams[key].append(word)
#  print anagrams["aeilnrt"]
  # alternatively use nltk.Index
  anagrams = nltk.Index(("".join(sorted(w)), w) for w in words)
  print anagrams["aeilnrt"]

def analysis_using_word_and_prev_pos():
  from nltk.corpus import brown
  pos = nltk.defaultdict(lambda: nltk.defaultdict(int))
  brown_news_tagged = brown.tagged_words(categories="news", simplify_tags=True)
  for ((w1,t1), (w2,t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1,w2)][t2] += 1
  print pos[("DET", "right")]

def invert_dictionary():
  counts = nltk.defaultdict(int)
  for word in nltk.corpus.gutenberg.words("milton-paradise.txt"):
    counts[word] += 1
  print [key for (key, value) in counts.items() if value == 32]

def tagging_tests():
  from nltk.corpus import brown
  brown_tagged_sents = brown.tagged_sents(categories="news")
  brown_sents = brown.sents(categories="news")
  # most frequent tag for this text
  tags = [tag for (word,tag) in brown.tagged_words(categories="news")]
  print "most frequent tag:", nltk.FreqDist(tags).max()
  # create default tagger with most frequent tag (NN)
  default_tagger = nltk.DefaultTagger("NN")
  print "evaluate(DefaultTagger)=", default_tagger.evaluate(brown_tagged_sents)
  # RegexpTagger allows POS tag patterns to be specified (p 199)
  # ...
  # Tagger based on knowing tags of top frequent words...
  fd = nltk.FreqDist(brown.words(categories="news"))
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
  most_freq_words = fd.keys()[:100]
  likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
  baseline_tagger = nltk.UnigramTagger(model=likely_tags)
  print "evaluate(UnigramTagger)=", baseline_tagger.evaluate(brown_tagged_sents)
  baseline_tagger2 = nltk.UnigramTagger(model=likely_tags,
    backoff=nltk.DefaultTagger("NN"))
  print "evaluate(baseline2)=", baseline_tagger2.evaluate(brown_tagged_sents)

def _evaluate_tagger(cfd, wordlist, sents):
  lt = dict((word, cfd[word].max()) for word in wordlist)
  tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger("NN"))
  return tagger.evaluate(sents)

def effect_of_model_size_on_tagger():
  from nltk.corpus import brown
  import pylab
  sents = brown.tagged_sents(categories="news")
  words_by_freq = list(nltk.FreqDist(brown.words(categories="news")))
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
  sizes = 2 ** pylab.arange(15)
  perfs = [_evaluate_tagger(cfd, words_by_freq[:size], sents) for size in sizes]
#  pylab.plot(sizes, perfs, "-bo")
  pylab.semilogx(sizes, perfs, "-bo")
  pylab.title("Lookup Tagger Performance with Varying Model Size")
  pylab.xlabel("Model Size")
  pylab.ylabel("Performance")
  pylab.show()

def train_and_test_tagger():
  from nltk.corpus import brown
  brown_tagged_sents = brown.tagged_sents(categories="news")
  size = int(len(brown_tagged_sents) * 0.9)
  train_sents = brown_tagged_sents[:size]
  test_sents = brown_tagged_sents[size:]
  unigram_tagger = nltk.UnigramTagger(train_sents)
  print unigram_tagger.evaluate(test_sents)

def show_sparse_data_problem_with_bigram_tagger():
  from nltk.corpus import brown
  brown_tagged_sents = brown.tagged_sents(categories="news")
  size = int(len(brown_tagged_sents) * 0.9)
  train_sents = brown_tagged_sents[:size]
  test_sents = brown_tagged_sents[size:]
  bigram_tagger = nltk.BigramTagger(train_sents)
  print "eval(train)=", bigram_tagger.evaluate(train_sents)
  print "eval(test)=", bigram_tagger.evaluate(test_sents)

def nested_backoff_tagger():
  from nltk.corpus import brown
  brown_tagged_sents = brown.tagged_sents(categories="news")
  size = int(len(brown_tagged_sents) * 0.9)
  train_sents = brown_tagged_sents[:size]
  test_sents = brown_tagged_sents[size:]
  t0 = nltk.DefaultTagger("NN")
  # unknown words handling can be improved by training the UnigramTagger
  # with the most frequent n words and POS, and making everything else an
  # POS UNK. So the ngram (prev n POS + current word) will be able to 
  # report POS of UNK words based on context better.
  t1 = nltk.UnigramTagger(train_sents, backoff=t0)
  t2 = nltk.BigramTagger(train_sents, backoff=t1)
  t3 = nltk.TrigramTagger(train_sents, backoff=t2)
  print t3.evaluate(test_sents)
  # store the taggers
  from cPickle import load, dump
  output = open("t3.pkl", "wb")
  dump(t3, output, -1)
  output.close()
  input = open("t3.pkl", "rb")
  t3_pickled = load(input)
  input.close()
  print t3_pickled.evaluate(test_sents)

def ambiguous_tags():
  from nltk.corpus import brown
  from cPickle import load
  brown_tagged_sents = brown.tagged_sents(categories="news")
  cfd = nltk.ConditionalFreqDist(
    ((x[1], y[1], z[0]), z[1])
    for sent in brown_tagged_sents
    for x, y, z in nltk.trigrams(sent))
  ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
  print "pct ambiguous=", sum(cfd[c].N() for c in ambiguous_contexts) * 100 / cfd.N()
  # study tagger's mistakes
  input = open("t3.pkl", "rb")
  t3 = load(input)
  input.close()
  test_tags = [tag for sent in brown.sents(categories="editorial")
                   for (word, tag) in t3.tag(sent)]
  gold_tags = [tag for (word, tag) in brown.tagged_words(categories="editorial")]
  print nltk.ConfusionMatrix(gold_tags, test_tags)
  
def main():
#  basic_tagger_usage()
#  similar_words()
#  tagged_token_representation()
#  common_verbs_in_news()

#  tagdict = findtags("NN", nltk.corpus.brown.tagged_words(categories="news"))
#  for tag in sorted(tagdict):
#   print tag, tagdict[tag]

#  how_is_often_used_in_text()
  
#  find_verb_to_verb_patterns()
#  find_highly_ambiguous_words()
#  tag_most_frequent_words()
#  word_count()
#  anagrams()
#  analysis_using_word_and_prev_pos()
#  invert_dictionary()
#  tagging_tests()
  effect_of_model_size_on_tagger() # warn: takes long
#  train_and_test_tagger()
#  show_sparse_data_problem_with_bigram_tagger()
#  nested_backoff_tagger()
#  ambiguous_tags()

  print "end"
  
if __name__ == "__main__":
  main()
