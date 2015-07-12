# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = "models"
NUM_TOPICS = 4

dtdf = pd.read_csv(os.path.join(MODELS_DIR, "doc_topics.csv"), sep="\t", 
                   names=["doc_id", "topic_id", "topic_prob"], 
                   skiprows=0)
# Choose 5 documents randomly for analysis
max_doc_id = dtdf["doc_id"].max()
doc_ids = []
for i in range(6):
    doc_ids.append(int(random.random() * max_doc_id))

for doc_id in doc_ids:
    filt = dtdf[dtdf["doc_id"] == doc_id]
    topic_ids = filt["topic_id"].tolist()
    topic_probs = filt["topic_prob"].tolist()
    prob_dict = dict(zip(topic_ids, topic_probs))
    ys = []
    for i in range(NUM_TOPICS):    
        if prob_dict.has_key(i):
           ys.append(prob_dict[i])
        else:
            ys.append(0.0)
    plt.title("Document #%d" % (doc_id))
    plt.ylabel("P(topic)")
    plt.ylim(0.0, 1.0)
    plt.xticks(range(NUM_TOPICS), 
               ["Topic#%d" % (x) for x in range(NUM_TOPICS)])
    plt.grid(True)
    plt.bar(range(NUM_TOPICS), ys, align="center")
    plt.show()
    
    