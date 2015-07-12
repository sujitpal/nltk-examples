# -*- coding: utf-8 -*-
import logging
import os
import gensim

def iter_docs(topdir):
    for f in os.listdir(topdir):
        fin = open(os.path.join(topdir, f), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in text.split(" "))

class MyBokCorpus(object):
    
    def __init__(self, topdir):
        self.topdir = topdir
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir))
        
    def __iter__(self):
        for tokens in iter_docs(self.topdir):
            yield self.dictionary.doc2bow(tokens)
            
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)
BOK_DIR = "/Users/palsujit/Projects/med_data/mtcrawler/kea_keys"
MODELS_DIR = "models"

corpus = MyBokCorpus(BOK_DIR)
tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

corpus.dictionary.save(os.path.join(MODELS_DIR, "bok.dict"))
gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "bok.mm"), 
                                  corpus_tfidf)