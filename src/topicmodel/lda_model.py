import logging
import os
import gensim

MODELS_DIR = "models"
NUM_TOPICS = 4

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, "bok.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "bok.mm"))

# Project to LDA space
lda = gensim.models.LdaModel(corpus, id2word=dictionary, 
                             iterations=300,
                             num_topics=NUM_TOPICS)

ftt = open(os.path.join(MODELS_DIR, "topic_terms.csv"), 'wb')
for topic_id in range(NUM_TOPICS):
    term_probs = lda.show_topic(topic_id, topn=50)
    for prob, term in term_probs:
       ftt.write("%d\t%s\t%.3f\n" % (topic_id, term.replace("_", " "), prob))
ftt.close()

fdt = open(os.path.join(MODELS_DIR, "doc_topics.csv"), 'wb')
for doc_id in range(len(corpus)):
    docbok = corpus[doc_id]
    doc_topics = lda.get_document_topics(docbok)
    for topic_id, topic_prob in doc_topics:
        fdt.write("%d\t%d\t%.3f\n" % (doc_id, topic_id, topic_prob))
fdt.close()
