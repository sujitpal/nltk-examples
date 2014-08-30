import logging
import os
import gensim

MODELS_DIR = "models"
NUM_TOPICS = 5

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, "mtsamples.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))

# Project to LDA space
lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
lda.print_topics(NUM_TOPICS)

