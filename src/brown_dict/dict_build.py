import nltk
from nltk.corpus import brown

DELIM = "_|_"
NORMED_TAGS = ["NN", "VB", "JJ", "RB", "DT", "IN", "OT"]
POSTAGS = {
    "NN" : "noun",
    "VB" : "verb",
    "JJ" : "adjective",
    "RB" : "adverb",
    "DT" : "determiner",
    "IN" : "preposition",
    "OT" : "other"
}

def normalize_brown_postags():
    brown_tags = open("../../data/brown_dict/brown_tags.csv", 'rb')
    tag_map = dict()
    for line in brown_tags:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        tag_name, tag_description = line.split("\t")[0:2]
        tag_desc_words = set(nltk.word_tokenize(tag_description.lower()))
        is_tagged = False
        for normed_tag in NORMED_TAGS[:-1]:
            desc_pattern = POSTAGS[normed_tag]
            if desc_pattern in tag_desc_words:
                tag_map[tag_name] = normed_tag
                is_tagged = True
        if not is_tagged:
            tag_map[tag_name] = "OT"
    brown_tags.close()
    return tag_map

def retag_brown_words(tag_map):
    wordpos_fd = nltk.FreqDist()
    for word, tag in brown.tagged_words():
        if tag_map.has_key(tag):
            normed_pos = tag_map[tag]
            retagged_word = DELIM.join([word.lower(), normed_pos])
            wordpos_fd.inc(retagged_word)  
    return wordpos_fd
    
def compose_record(word, wordpos_fd):
    freqs = []
    for tag in NORMED_TAGS:
        wordpos = DELIM.join([word, tag])
        freqs.append(wordpos_fd[wordpos])
    sum_freqs = float(sum(freqs))
    nf = [float(f) / sum_freqs for f in freqs]
    return "%s\t%s\n" % (word, "\t".join(["%5.3f" % (x) for x in nf]))


tag_map = normalize_brown_postags()
wordpos_fd = retag_brown_words(tag_map)
already_seen_words = set()
brown_dict = open("../../data/brown_dict/brown_dict.csv", 'wb')
brown_dict.write("#WORD\t%s\n" % ("\t".join(NORMED_TAGS)))
for wordpos in wordpos_fd.keys():
    word, tag = wordpos.split(DELIM)
    if word in already_seen_words:
        continue
    brown_dict.write(compose_record(word, wordpos_fd))
    already_seen_words.add(word)
brown_dict.close()
