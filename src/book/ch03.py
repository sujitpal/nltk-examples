#!/usr/bin/python
# Processing raw text

from __future__ import division
import nltk 
import re
import pprint
import urllib2
import feedparser
import codecs

def download(url, file):
  req = urllib2.Request(url)
  req.add_header("User-Agent", "Mozilla/5.0 (Windows; U; Windows NT 5.1; es-ES; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5")
  raw = urllib2.urlopen(req).read()
  f = open(file, 'w')
  f.write(raw)
  f.close()

def web_file_plain():
#  download("http://www.gutenberg.org/files/2554/2554.txt", "/tmp/2554.txt")
  f = open("/tmp/2554.txt", 'r')
  raw = f.read()
  f.close()
  print "raw", type(raw), len(raw), raw[:75]
  tokens = nltk.word_tokenize(raw)
  print "tokens", type(tokens), len(tokens), tokens[:10]
  text = nltk.Text(tokens)
  print "text[1020:1060", text[1020:1060]
  print "colloc=", text.collocations()
  start = raw.find("PART I")
  end = raw.rfind("End of Project Gutenberg's Crime")
  raw2 = raw[start:end]
  print "index(PART I)=", raw.find("PART I"), raw2.find("PART I")

def web_file_html():
#  download("http://news.bbc.co.uk/2/hi/health/2284783.stm", "/tmp/2284783.stm")
  f = open("/tmp/2284783.stm", 'r')
  html = f.read()
  f.close()
  raw = nltk.clean_html(html)
  tokens = nltk.word_tokenize(raw)
  text = nltk.Text(tokens[96:399])
  text.concordance("gene")

def web_file_rss():
  download("http://languagelog.ldc.upenn.edu/nll/?feed=atom",
    "/tmp/feed.xml")
  f = open("/tmp/feed.xml", 'r')
  llog = feedparser.parse(f.read())
  print "title,len(content)=", llog["feed"]["title"], len(llog.entries)
  post = llog.entries[2]
  content = post.content[0].value
  print "title,countent...=", post.title, content[:70]
  tokens = nltk.word_tokenize(nltk.clean_html(content))
  print "tokens=", tokens

def unicode_read():
  path = "/opt/nltk_data/corpora/unicode_samples/polish-lat2.txt"
  f = codecs.open(path, encoding="latin2")
  for line in f:
    print line.strip().encode("unicode_escape")

def basic_regexps():
  wordlist = [w for w in nltk.corpus.words.words("en") if w.islower()]
  print "words ending with -ed", [w for w in wordlist if re.search("ed$", w)]
  print "crossword pattern", [w for w in wordlist if re.search("^..j..t..$", w)]
  print "textonyms(golf)=", [w for w in wordlist if re.search("^[ghi][mno][jlk][def]$", w)]
  chat_words = sorted(set([w for w in nltk.corpus.nps_chat.words()]))
  print "mine=", [w for w in chat_words if re.search("^m+i+n+e+$", w)]

def compress(regex, word):
  pieces = re.findall(regex, word)
  return "".join(pieces)

def compress_vowels():
  # initial vowel sequence, final vowel sequence or consonents,
  # everything else is removed
  regex = r"^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]"
  english_udhr = nltk.corpus.udhr.words("English-Latin1")
  print nltk.tokenwrap([compress(regex, w) for w in english_udhr[:75]])

def consonant_vowel_sequences_rotokas():
  rotokas_words = nltk.corpus.toolbox.words("rotokas.dic")
  cvs = [cv for w in rotokas_words
    for cv in re.findall(r"[ptksvr][aeiou]", w)]
  cfd = nltk.ConditionalFreqDist(cvs)
  cfd.tabulate()
  cv_word_pairs = [(cv, w) for w in rotokas_words
                           for cv in re.findall(r"[ptksrv][aeiou]", w)]
  cv_index = nltk.Index(cv_word_pairs)
  print "index(su)=", cv_index["su"]
  print "index(po)=", cv_index["po"]

def discover_hypernyms():
  from nltk.corpus import brown
  text = nltk.Text(brown.words(categories=["hobbies", "learned"]))
#  print text.findall(r"<\w*> <and> <other> <\w*>")
  print text.findall(r"<as> <\w*> <as> <\w*>")

def find_in_stemmed_index(word):
#  porter = nltk.PorterStemmer()
  wnl = nltk.WordNetLemmatizer()
  grail = nltk.corpus.webtext.words("grail.txt")
#  index = nltk.Index([(porter.stem(w.lower()), pos)
#    for (pos, w) in enumerate(grail)])
  index = nltk.Index([(wnl.lemmatize(w.lower()), pos)
    for (pos, w) in enumerate(grail)])
  for pos in index[word]:
    lcontext = " ".join(grail[pos-4:pos])
    rcontext = " ".join(grail[pos:pos+4])
    print lcontext, rcontext

def regex_word_tokenize():
  # developing own tokenizer, compare between
  # nltk.corpus.treebank_raw.raw() and
  # nltk.corpus.treebank.words()
  alice = nltk.corpus.gutenberg.raw("carroll-alice.txt")
#  print re.split(r" ", alice)
#  print re.split(r"\W+", alice) # split on any non-word not only space
#  print re.findall(r"\w+|\S\w*", alice) # seq of 2/more punct separated
#  print re.findall(r"\w+(?:[-']\w)*|'|[-.(]+\s\w*", alice)
  pattern = r"""(?x)     # verbose regexp
    ([A-Z]\.)+ |         # abbreviations (U.S.A.)
    \w+(-\w+)* |         # words with optional internal hyphens
    \$?\d+(\.\d+)?%? |   # currency and percentages
    \.\.\. |             # ellipsis
    [][.,;"'?():-_`]     # separator tokens
  """
  print nltk.regexp_tokenize(alice, pattern)

def sentence_tokenization():
  sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
  text = nltk.corpus.gutenberg.raw("chesterton-thursday.txt")
  sents = sent_tokenizer.tokenize(text)
  pprint.pprint(sents[171:181])
  
def main():
#  web_file_plain()
#  web_file_html()
#  web_file_rss()
#  unicode_read()
#  basic_regexps()
#  compress_vowels()
#  consonant_vowel_sequences_rotokas()
#  discover_hypernyms()
#  find_in_stemmed_index("offic") # porter
#  find_in_stemmed_index("officer") # wordnet
#  regex_word_tokenize()
#  sentence_tokenization()
  pass
  
if __name__ == "__main__":
  main()

