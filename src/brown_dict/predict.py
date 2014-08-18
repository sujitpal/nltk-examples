from __future__ import division
import numpy as np
import nltk

NORMTAGS = ["NN", "VB", "JJ", "RB", "DT", "IN", "OT"]

def load_word_dict(dict_file):
    word_dict = {}
    fdict = open(dict_file, "rb")
    for line in fdict:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        cols = line.split("\t")
        word = cols[0]
        probs = [float(x) for x in cols[1:]]
        word_dict[word] = probs
    fdict.close()
    return word_dict

def load_phrase_tags(phrase_tag_file):
    phrase_tags = set()
    ftags = open(phrase_tag_file, 'rb')
    for line in ftags:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        phrase, count = line.split("\t")
        phrase_tags.add(phrase)
    ftags.close()
    return phrase_tags

def assert_true(fn, message):
    if fn != True:
        print "Assert failed:", message

def tag_to_index(tag):
    if tag == "START":
        return 0
    elif tag == "END":
        return len(NORMTAGS) + 1
    else:
        return NORMTAGS.index(tag) + 1

def index_to_tag(index):
    if index == 0: 
        return "START"
    elif index == len(NORMTAGS) + 1:
        return "END"
    else:
        return NORMTAGS[index - 1]

def predict_likely_pos(prev_tag, trans_probs):
    row = tag_to_index(prev_tag)
    probs = trans_probs[row, :]
    return index_to_tag(np.argmax(probs))

def predict_pos(word, word_dict):
    if word_dict.has_key(word):
        probs = np.array(word_dict[word])
        return NORMTAGS[np.argmax(probs)]
    else:
        return "OT"
        
def predict_if_noun(word, word_dict):
    return predict_pos(word, word_dict) == "NN"

def predict_if_noun_phrase(phrase, trans_probs, phrase_tags):
    words = nltk.word_tokenize(phrase)
    tags = []
    for word in words:
        if word_dict.has_key(word):
            tags.append(predict_pos(word, word_dict))
        else:
            prev_tag = "START" if len(tags) == 0 else tags[-1]
            tags.append(predict_likely_pos(prev_tag, trans_probs))
    return " ".join(tags) in phrase_tags

# test cases for individual words
word_dict = load_word_dict("../../data/brown_dict/brown_dict.csv")
assert_true(predict_if_noun("hypothalamus", word_dict), "Hypothalamus == NOUN!")
assert_true(not predict_if_noun("intermediate", word_dict), "Intermediate != NOUN!")
assert_true(predict_if_noun("laugh", word_dict), "Laugh ~= NOUN!")

# test cases for phrases
phrase_tags = load_phrase_tags("../../data/brown_dict/np_tags.csv")
trans_probs = np.loadtxt("../../data/brown_dict/pos_trans.csv", delimiter="\t")
assert_true(predict_if_noun_phrase("time flies", trans_probs, phrase_tags), 
            "time flies == NP!")
assert_true(not predict_if_noun_phrase("were spoken", trans_probs, phrase_tags), 
            "were spoken == VP!")
            
