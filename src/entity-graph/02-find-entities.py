import os
import spacy

DATA_DIR = "../../data/entity-graph"
SENTENCES_FILENAME = os.path.join(DATA_DIR, "sentences.tsv")

ENTITIES_FILENAME = os.path.join(DATA_DIR, "entities.tsv")

nlp = spacy.load("en")

num_sents, num_ents = 0, 0
fents = open(ENTITIES_FILENAME, "w")
fsent = open(SENTENCES_FILENAME, "r")
for line in fsent:
    if num_sents % 100 == 0:
        print("{:d} entities found in {:d} sentences".format(num_ents, num_sents))
    pid, sid, sent = line.strip().split('\t')
    doc = nlp(sent)
    for ent in doc.ents:
        fents.write("{:d}\t{:s}\t{:s}\t{:s}\t{:d}\t{:d}\n".format(
            int(sid), sent, ent.text, ent.label_, ent.start_char, ent.end_char))
        num_ents += 1
    num_sents += 1

print("{:d} entities found in {:d} sentences, COMPLETE".format(num_ents, num_sents))

fsent.close()
fents.close()
