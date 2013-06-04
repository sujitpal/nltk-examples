from __future__ import division
from mrjob.job import MRJob
import nltk
import string

class NGramCountingJob(MRJob):

  def mapper_init(self):
#    self.stopwords = nltk.corpus.stopwords.words("english")
    self.stopwords = set(['i', 'me', 'my', 'myself', 'we',
      'our', 'ours', 'ourselves', 'you', 'your', 'yours',
      'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
      'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
      'they', 'them', 'their', 'theirs', 'themselves', 'what',
      'which', 'who', 'whom', 'this', 'that', 'these', 'those',
      'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'having', 'do', 'does', 'did',
      'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
      'because', 'as', 'until', 'while', 'of', 'at', 'by',
      'for', 'with', 'about', 'against', 'between', 'into',
      'through', 'during', 'before', 'after', 'above', 'below',
      'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
      'over', 'under', 'again', 'further', 'then', 'once',
      'here', 'there', 'when', 'where', 'why', 'how', 'all',
      'any', 'both', 'each', 'few', 'more', 'most', 'other',
      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
      'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
      'just', 'don', 'should', 'now'])
    self.porter = nltk.PorterStemmer()

  def mapper(self, key, value):

    def normalize_numeric(x):
      xc = x.translate(string.maketrans("", ""), string.punctuation)
      return "_NNN_" if xc.isdigit() else x

    def normalize_stopword(x):
      return "_SSS_" if str(x) in self.stopwords else x

    cols = value.split("|")
    words = nltk.word_tokenize(cols[1])
    # normalize number and stopwords and stem remaining words
    words = [word.lower() for word in words]
    words = [normalize_numeric(word) for word in words]
    words = [normalize_stopword(word) for word in words]
    words = [self.porter.stem(word) for word in words]
    trigrams = nltk.trigrams(words)
    for trigram in trigrams:
      yield (trigram, 1)
      bigram = trigram[1:]
      yield (bigram, 1)
      unigram = bigram[1:]
      yield (unigram, 1)

  def reducer(self, key, values):
    yield (key, sum([value for value in values]))

if __name__ == "__main__":
  NGramCountingJob.run()
  