# -*- coding: utf-8 -*-
import nltk
import os

USER_KEYWORDS_DIR = "/Users/palsujit/Projects/med_data/mtcrawler/kea/test"
KEA_KEYWORDS_DIR = "/Users/palsujit/Projects/med_data/mtcrawler/kea/test/keys"
KEYWORDS_FILE = "/Users/palsujit/Projects/med_data/mtcrawler/kea/merged_keys.txt"
CUSTOM_STOPWORDS = ["patient", "normal", "mg"]

def main():

    # get set of english keywords from NLTK
    stopwords = set(nltk.corpus.stopwords.words("english"))
    # add own corpus-based stopwords based on high IDF words
    for custom_stopword in CUSTOM_STOPWORDS:
        stopwords.add(custom_stopword)
        
    keywords = set()
    for f in os.listdir(USER_KEYWORDS_DIR):
        # only select the .key files
        if f.endswith(".txt") or f == "keys":
            continue
        fusr = open(os.path.join(USER_KEYWORDS_DIR, f), 'rb')
        for line in fusr:
            line = line.strip().lower()
            if line in keywords:
                continue
            keywords.add(line)
        fusr.close()
    for f in os.listdir(KEA_KEYWORDS_DIR):
        fkea = open(os.path.join(KEA_KEYWORDS_DIR, f), 'rb')
        for line in fkea:
            keywords.add(line.strip())
        fkea.close()
    fmrg = open(KEYWORDS_FILE, 'wb')
    for keyword in keywords:
        if keyword in stopwords:
            continue
        fmrg.write("%s\n" % (keyword))
    fmrg.close()

if __name__ == "__main__":
    main()
