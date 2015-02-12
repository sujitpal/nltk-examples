# -*- coding: utf-8 -*-
import drug_ner_utils as dnu
import os

def build_ngram_text(infile, outfile):
    fin = open(os.path.join(dnu.DATA_DIR, infile), 'rb')
    fout = open(os.path.join(dnu.DATA_DIR, outfile), 'wb')
    for line in fin:
        for word in line.strip().split():
            ngrams = dnu.str_to_ngrams(word, dnu.GRAM_SIZE)
            if len(ngrams) > 0:
                fout.write("%s\n" % " ".join(ngrams))
    fin.close()
    fout.close()


build_ngram_text("generic_names.txt", "generic_positive.txt")
build_ngram_text("brand_names.txt", "brand_positive.txt")
build_ngram_text("raw_data.txt", "unlabeled.txt")
