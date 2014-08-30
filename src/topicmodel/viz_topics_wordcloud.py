import os
import wordcloud

MODELS_DIR = "models"

final_topics = open(os.path.join(MODELS_DIR, "final_topics.txt"), 'rb')
curr_topic = 0
for line in final_topics:
    line = line.strip()[line.rindex(":") + 2:]
    scores = [float(x.split("*")[0]) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))
    elements = wordcloud.fit_words(freqs, width=120, height=120)
    wordcloud.draw(elements, "gs_topic_%d.png" % (curr_topic),
                   width=120, height=120)
    curr_topic += 1
final_topics.close()
