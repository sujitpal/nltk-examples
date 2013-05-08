from __future__ import division

from operator import itemgetter

import nltk.cluster.util as nltkutil
import numpy as np
import random
import re
import scam_dist as scam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

def preprocess(fnin, fnout):
  fin = open(fnin, 'rb')
  fout = open(fnout, 'wb')
  buf = []
  id = ""
  category = ""
  for line in fin:
    line = line.strip()
    if line.find("-- Document Separator --") > -1:
      if len(buf) > 0:
        # write out body,
        body = re.sub("\s+", " ", " ".join(buf))
        fout.write("%s\t%s\t%s\n" % (id, category, body))
      # process next header and init buf
      id, category, rest = map(lambda x: x.strip(), line.split(": "))
      buf = []
    else:
      # process body
      buf.append(line)
  fin.close()
  fout.close()

def train(fnin):
  docs = []
  cats = []
  fin = open(fnin, 'rb')
  for line in fin:
    id, category, body = line.strip().split("\t")
    docs.append(body)
    cats.append(category)
  fin.close()
  pipeline = Pipeline([
    ("vect", CountVectorizer(min_df=0, stop_words="english")),
    ("tfidf", TfidfTransformer(use_idf=False))])
  tdMatrix = pipeline.fit_transform(docs, cats)
  return tdMatrix, cats

def test(tdMatrix, cats, fsim):
  testIds = random.sample(range(0, len(cats)), int(0.1 * len(cats)))
  testIdSet = set(testIds)
  refIds = filter(lambda x: x not in testIdSet, range(0, len(cats)))
  sims = np.zeros((len(testIds), len(refIds)))
  for i in range(0, len(testIds)):
    for j in range(0, len(refIds)):
      doc1 = np.asarray(tdMatrix[testIds[i], :].todense()).reshape(-1)
      doc2 = np.asarray(tdMatrix[refIds[j], :].todense()).reshape(-1)
      sims[i, j] = fsim(doc1, doc2)
  for i in range(0, sims.shape[0]):
    xsim = list(enumerate(sims[i, :]))
    sortedSims = sorted(xsim, key=itemgetter(1), reverse=True)[0:5]
    sourceCat = cats[testIds[i]]
    numMatchedCats = 0
    numTestedCats = 0
    for j, score in sortedSims:
      targetCat = cats[j]
      if sourceCat == targetCat:
        numMatchedCats += 1
      numTestedCats += 1
    print("Test Doc: %d, Source Category: %s, Target Matched: %d/%d times" %
      (i, sourceCat, numMatchedCats, numTestedCats))
      
def main():
  preprocess("sugar-coffee-cocoa-docs.txt", "sccpp.txt")
  tdMatrix, cats = train("sccpp.txt")
  print "Results with Cosine Distance Similarity Measure"
  test(tdMatrix, cats, nltkutil.cosine_distance)
  print "Results with Euclidean Distance Similarity Measure"
  test(tdMatrix, cats, nltkutil.euclidean_distance)
  print "Results with SCAM Distance Similarity Measure"
  test(tdMatrix, cats, scam.scam_distance)
  
if __name__ == "__main__":
  main()
  