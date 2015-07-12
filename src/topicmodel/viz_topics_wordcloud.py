import matplotlib.pyplot as plt
import os
import pandas as pd
import wordcloud

MODELS_DIR = "models"

ttdf = pd.read_csv(os.path.join(MODELS_DIR, "topic_terms.csv"), 
                   sep="\t", skiprows=0, names=["topic_id", "term", "prob"])
topics = ttdf.groupby("topic_id").groups
for topic in topics.keys():
    row_ids = topics[topic]
    freqs = []
    for row_id in row_ids:
        row = ttdf.ix[row_id]
        freqs.append((row["term"], row["prob"]))
    wc = wordcloud.WordCloud()
    elements = wc.fit_words(freqs)
    plt.figure(figsize=(5, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()