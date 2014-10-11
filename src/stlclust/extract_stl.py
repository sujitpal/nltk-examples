import os
import nltk
import string
import re
from operator import itemgetter

INPUT_DIR = "/home/sujit/Projects/med_data/mtcrawler/texts"
OUTPUT_DIR = "../../data/stlclust"
PUNCTUATIONS = set([c for c in string.punctuation])
DIGITS = set([c for c in string.digits])
BULLETS = re.compile("[0-9IVXA-Za-z]{0,3}\.")
PUNCTS = re.compile(r"[" + string.punctuation + "]")

def find_first(line, cs):
    idxs = []
    for c in cs:
        c_index = line.find(c)
        if c_index > -1:
            # if this occurs after an existing punctuation, then discard
            prev_chars = set([pc for pc in line[0:c_index - 1]])
            if len(PUNCTUATIONS.intersection(prev_chars)) > 0:
                return -1
            # make sure this position is either EOL or followed by space
            if c_index + 1 == len(line) or line[c_index + 1] == ' ':
                idxs.append(c_index)
    if len(idxs) == 0:
        return -1
    else:
        return min(idxs)
        
stfd = nltk.FreqDist()
for filename in os.listdir(INPUT_DIR):
    f = open(os.path.join(INPUT_DIR, filename), 'rb')
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        # Isolate section titles from text. Titles are leading phrases 
        # terminated by colon or hyphen. Usually all-caps but can be in
        # mixed-case also
        sec_title = None
        corh = find_first(line, [":", "-"])
        if corh > -1:
            sec_title = line[0:corh]
        # Alternatively, if the line is all caps, then it is also a
        # section title
        if sec_title is None and line.upper() == line:
            sec_title = line
        if sec_title is not None: 
            # Remove retrieved titles with leading arabic number, roman number
            # and alpha bullets (allow max 3) bullets
            if re.match(BULLETS, sec_title) is not None:
                continue
            # Remove sections that look like dates (all numbers once puncts)
            # are removed
            if re.sub(PUNCTS, "", sec_title).isdigit():
                continue
            # if retrieved title is mixed case remove any that have > 4 words
            if sec_title != sec_title.upper() and len(sec_title.split()) > 4:
                continue
            # if retrieved title contains special chars, remove
            if "," in sec_title:
                continue
            # replace "diagnoses" with "diagnosis"
            sec_title = re.sub("DIAGNOSES", "DIAGNOSIS", sec_title)
            stfd[sec_title] += 1
    f.close()
    
# output the frequency distribution
fout = open(os.path.join(OUTPUT_DIR, "stitles.txt"), 'wb')
for k, v in sorted(stfd.items(), key=itemgetter(1), reverse=True):
    fout.write("%s\t%d\n" % (k, v))
fout.close()
    