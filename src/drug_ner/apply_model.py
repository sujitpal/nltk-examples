from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import drug_ner_utils as dnu
import numpy as np
import os

def vectorize_ngrams(ngrams, vocab):
    vec = np.zeros((1, len(vocab)))
    for ngram in ngrams:
        if vocab.has_key(ngram):
            vec[0, vocab[ngram]] = 1
    return vec
    

X, y, generic_vec = dnu.vectorize("unlabeled.txt", "generic_positive.txt", 100)
y = joblib.load(os.path.join(dnu.DATA_DIR, "y_generic_4.pkl"))
generic_clf = LinearSVC()
generic_clf.fit(X, y)
print("Score for generic classifier: %.3f" % (generic_clf.score(X, y)))

X, y, brand_vec = dnu.vectorize("unlabeled.txt", "brand_positive.txt", 100)

y = joblib.load(os.path.join(dnu.DATA_DIR, "y_brand_3.pkl"))
brand_clf = LinearSVC()
brand_clf.fit(X, y)
print("Score for brand classifier: %.3f" % (brand_clf.score(X, y)))

fraw = open(os.path.join(dnu.DATA_DIR, "raw_data.txt"), 'rb')
i = 0
for line in fraw:
    line = line.strip().lower()
    annotated = []
    for word in line.split():
        ngrams = dnu.str_to_ngrams(word, dnu.GRAM_SIZE)
        Xgen = generic_vec.transform([" ".join(ngrams)])
        Xbrand = brand_vec.transform([" ".join(ngrams)])
        is_generic = generic_clf.predict(Xgen)
        is_brand = brand_clf.predict(Xbrand)
        if is_generic == 1:
            annotated.append("<GENERIC>" + word + "</GENERIC>")
        elif is_brand == 1:
            annotated.append("<BRAND>" + word + "</BRAND>")
        else:
            annotated.append(word)
    print("Input: %s" % (line))
    print("Output: %s" % (" ".join(annotated)))
    i += 1
    if i > 10:
        break
fraw.close()
