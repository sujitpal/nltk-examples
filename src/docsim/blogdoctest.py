import numpy as np
import re
import scam_dist as scam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

def main():
  docs = []
  cats = []
  files = ["blogdocs/doc1.txt", "blogdocs/doc2.txt", "blogdocs/doc3.txt"]
  for file in files:
    f = open(file, 'rb')
    body = re.sub("\\s+", " ", " ".join(f.readlines()))
    f.close()
    docs.append(body)
    cats.append("X")
  pipeline = Pipeline([
    ("vect", CountVectorizer(min_df=0, stop_words="english")),
    ("tfidf", TfidfTransformer(use_idf=False))])
  tdMatrix = pipeline.fit_transform(docs, cats)
  testDocs = []
  for i in range(0, tdMatrix.shape[0]):
    testDocs.append(np.asarray(tdMatrix[i, :].todense()).reshape(-1))
  scamDist12 = scam.scam_distance(testDocs[0], testDocs[1])
  scamDist23 = scam.scam_distance(testDocs[1], testDocs[2])
  print scamDist12, scamDist23

if __name__ == "__main__":
  main()
