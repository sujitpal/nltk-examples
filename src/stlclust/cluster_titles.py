import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.io import mmread

OUTPUT_DIR = "../../data/stlclust"

X = mmread(os.path.join(OUTPUT_DIR, "stitles.mtx"))
clust = DBSCAN(eps=0.1, min_samples=5, metric="precomputed")
clust.fit(X)

# print cluster report
stitles = []
ftitles = open(os.path.join(OUTPUT_DIR, "stitles.txt"), 'rb')
for line in ftitles:
    stitles.append(line.strip().split("\t")[0])
ftitles.close()

preds = clust.labels_
clabels = np.unique(preds)
for i in range(clabels.shape[0]):
    if clabels[i] < 0:
        continue
    cmem_ids = np.where(preds == clabels[i])[0]
    cmembers = []
    for cmem_id in cmem_ids:
        cmembers.append(stitles[cmem_id])
    print "Cluster#%d: %s" % (i, ", ".join(cmembers))

    