# -*- coding: utf-8 -*-
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import string

DATA_DIR = "../../data/drug_ner"
GRAM_SIZE = 3
PUNCTS = set([c for c in string.punctuation])
NUMBERS = set([c for c in "0123456789"])

def is_punct(c):
    return c in PUNCTS
    
def is_number(c):
    return c in NUMBERS
    
def str_to_ngrams(instring, gram_size):
    ngrams = []
    for word in nltk.word_tokenize(instring.lower()):
        try:
            word = "".join(["S", word, "E"]).encode("utf-8")
            cword = [c for c in word if not(is_punct(c) or is_number(c))]
            ngrams.extend(["".join(x) for x in nltk.ngrams(cword, gram_size)])
        except UnicodeDecodeError:
            pass
    return ngrams
            
def ngram_distrib(names, gram_size):
    tokens = []
    for name in names:
        tokens.extend(str_to_ngrams(name, gram_size))
    return nltk.FreqDist(tokens)

def plot_ngram_distrib(fd, nbest, title, gram_size):
    kvs = sorted([(k, fd[k]) for k in fd], key=itemgetter(1), reverse=True)[0:nbest]
    ks = [k for k, v in kvs]
    vs = [v for k, v in kvs]
    plt.plot(np.arange(nbest), vs)
    plt.xticks(np.arange(nbest), ks, rotation="90")
    plt.title("%d-gram frequency for %s names (Top %d)" % 
              (gram_size, title, nbest))
    plt.xlabel("%d-grams" % (gram_size))
    plt.ylabel("Frequency")
    plt.show()

def truncate_fd(fd, nbest):
    kvs = sorted([(k, fd[k]) for k in fd], key=itemgetter(1), reverse=True)[0:nbest]
    return {k:v for k, v in kvs}

def vectorize(ufile, pfile, max_feats):
    text = []
    labels = []
    fno = 0
    for fname in [ufile, pfile]:
        f = open(os.path.join(DATA_DIR, fname), 'rb')
        for line in f:
            text.append(line.strip())
            labels.append(fno)
        fno = fno + 1
        f.close()
    vec = CountVectorizer(min_df=0.0, max_features=max_feats, binary=True)
    X = vec.fit_transform(text)
    y = np.array(labels)
    return X, y, vec
