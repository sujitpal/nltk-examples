from __future__ import division
import nltk
import numpy as np

from nltk.corpus import treebank_chunk

NORMTAGS = ["NN", "VB", "JJ", "RB", "DT", "IN", "OT"]
POSTAGS = {
    "NN" : "noun",
    "VB" : "verb",
    "JJ" : "adjective",
    "RB" : "adverb",
    "DT" : "determiner",
    "IN" : "preposition",
    "OT" : "other"
}

def normalize_ptb_tags():
    tag_map = {}
    ptb_tags = open("../../data/brown_dict/ptb_tags.csv", 'rb')
    for line in ptb_tags:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        tag, desc = line.split("\t")
        desc_words = nltk.word_tokenize(desc.lower().replace("-", " "))
        is_tagged = False
        for key in NORMTAGS[:-1]:
            postag_desc = POSTAGS[key]
            if postag_desc in desc_words:
                tag_map[tag] = key
                is_tagged = True
        if not is_tagged:
            tag_map[tag] = "OT"
    ptb_tags.close()
    return tag_map
    
def get_chunks(tree, phrase_type, tags):
    try:
        tree.node
    except AttributeError:
        return 
    else:
        if tree.node == phrase_type:
            tags.append(tree)
        else:
            for child in tree:
                get_chunks(child, phrase_type, tags)

def index_of(tag):
    if tag == "START":
        return 0
    elif tag == "END":
        return len(NORMTAGS) + 1
    else:
        return NORMTAGS.index(tag) + 1
        
def update_trans_freqs(trans_freqs, tag_seq):
    tags = ["START"]
    tags.extend(tag_seq.split(" "))
    tags.append("END")
    bigrams = nltk.bigrams(tags)
    for bigram in bigrams:
        row = index_of(bigram[0])
        col = index_of(bigram[1])
        trans_freqs[row, col] += 1
    
# generate phrases as a sequence of (normalized) POS tags and
# transition probabilities across POS tags.
tag_map = normalize_ptb_tags()
np_fd = nltk.FreqDist()
trans_freqs = np.zeros((len(NORMTAGS) + 2, len(NORMTAGS) + 2))
for tree in treebank_chunk.chunked_sents():
    chunks = []
    get_chunks(tree, "NP", chunks)
    for chunk in chunks:
        tagged_poss = [tagged_word[1] for tagged_word in chunk]
        normed_tags = []
        for tagged_pos in tagged_poss:
            try:
                normed_tags.append(tag_map[tagged_pos])
            except KeyError:
                normed_tags.append("OT")
        np_fd.inc(" ".join(normed_tags))
        
fout = open("../../data/brown_dict/np_tags.csv", 'wb')
for tag_seq in np_fd.keys():
    fout.write("%s\t%d\n" % (tag_seq, np_fd[tag_seq]))
    update_trans_freqs(trans_freqs, tag_seq)
fout.close()
# normalize so they are all probablities (by row sum)
trans_probs = trans_freqs / np.linalg.norm(trans_freqs, axis=1)[:, np.newaxis]
trans_probs[~np.isfinite(trans_probs)] = 0.0
np.savetxt("../../data/brown_dict/pos_trans.csv", trans_probs, fmt="%7.5f", delimiter="\t")
