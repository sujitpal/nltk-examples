from __future__ import division
import os
import numpy as np
from scipy.io import mmwrite
from fuzzywuzzy import fuzz

OUTPUT_DIR = "../../data/stlclust"

def compute_similarity(s1, s2):
    return 1.0 - (0.01 * max(
        fuzz.ratio(s1, s2),
        fuzz.token_sort_ratio(s1, s2),
        fuzz.token_set_ratio(s1, s2)))
        

cutoff = 2
stitles = []
fin = open(os.path.join(OUTPUT_DIR, "stitles.txt"), 'rb')
for line in fin:
    stitle, count = line.strip().split("\t")
    if int(count) < cutoff:
        continue
    stitles.append(stitle)
fin.close()

X = np.zeros((len(stitles), len(stitles)))
for i in range(len(stitles)):
    if i > 0 and i % 10 == 0:
        print "Processed %d/%d rows of data" % (i, X.shape[0])
    for j in range(len(stitles)):
        if X[i, j] == 0.0:        
            X[i, j] = compute_similarity(stitles[i].lower(), stitles[j].lower())
            X[j, i] = X[i, j]

# write to Matrix Market format for passing to DBSCAN
mmwrite(os.path.join(OUTPUT_DIR, "stitles.mtx"), X)

