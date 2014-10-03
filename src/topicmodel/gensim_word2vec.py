import string
import nltk
import numpy as np
from cStringIO import StringIO
from gensim.models import word2vec
import logging
logging.basicConfig(format="%(asctime)s: %(levelname)s : %(message)s", 
                    level=logging.INFO)

# load data
fin = open("/home/sujit/Projects/scalcium/src/main/resources/langmodel/raw_sentences.txt", 'rb')
puncts = set([c for c in string.punctuation])
sentences = []
for line in fin:
    # each sentence is a list of words, we lowercase and remove punctuations
    # same as the Scala code
    sentences.append([w for w in nltk.word_tokenize(line.strip().lower()) 
            if w not in puncts])
fin.close()

# train word2vec with sentences
model = word2vec.Word2Vec(sentences, size=100, window=4, min_count=1, workers=4)
model.init_sims(replace=True)

# find 10 words closest to "day"
print "words most similar to 'day':"
print model.most_similar(positive=["day"], topn=10)

# find closest word to "he"
print "words most similar to 'he':"
print model.most_similar(positive=["he"], topn=1)

# for each word in the vocabulary, write out the word vectors to a file
fvec = open("/tmp/word_vectors.txt", 'wb')
for word in model.vocab.keys():
    vec = model[word]
    for i in range(vec.shape[0]):
    s = StringIO()
    np.savetxt(s, vec, fmt="%.5f", newline=",")
    fvec.write("%s%s\n" % (s.getvalue(), word))
fvec.close()
