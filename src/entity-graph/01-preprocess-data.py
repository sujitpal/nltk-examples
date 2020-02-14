import os
import re
import spacy

DATA_DIR = "../../data/entity-graph"

TEXT_FILENAME = os.path.join(DATA_DIR, "db-article.txt")
ACRONYMS_FILENAME = os.path.join(DATA_DIR, "db-acronyms.txt")

SENTENCES_FILENAME = os.path.join(DATA_DIR, "sentences.tsv")

acronyms_lookup = dict()
facro = open(ACRONYMS_FILENAME, "r")
for line in facro:
    acro, full = line.strip().split('\t')
    acronyms_lookup[acro] = full

facro.close()

lm = spacy.load("en")

pid, sid = 0, 0
fsents = open(SENTENCES_FILENAME, "w")
ftext = open(TEXT_FILENAME, "r")
for para in ftext:
    para = para.strip()
    if len(para) == 0:
        continue
    for sent in lm(para).sents:
        if sid % 100 == 0:
            print("Wrote {:d} sents from {:d} paragraphs".format(sid, pid))
        sent_tokens = []
        for token in lm(sent.text):
            token_text = token.text
            if token_text in acronyms_lookup.keys():
                sent_tokens.append(acronyms_lookup[token_text])
            else:
                sent_tokens.append(token_text)
        fsents.write("{:d}\t{:d}\t{:s}\n".format(pid, sid, " ".join(sent_tokens)))
        sid += 1
    pid += 1

print("Wrote {:d} sents from {:d} paragraphs, COMPLETE".format(sid, pid))

ftext.close()
fsents.close()

