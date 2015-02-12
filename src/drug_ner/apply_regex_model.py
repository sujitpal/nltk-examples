from sklearn.externals import joblib
import drug_ner_utils as dnu
import os

generic_fd = set(dnu.truncate_fd(joblib.load(os.path.join(dnu.DATA_DIR, 
                                            "generic_fd.pkl")), 100))
brand_fd = set(dnu.truncate_fd(joblib.load(os.path.join(dnu.DATA_DIR, 
                                            "brand_fd.pkl")), 50))

fraw = open(os.path.join(dnu.DATA_DIR, "raw_data.txt"), 'rb')
i = 0
for line in fraw:
    line = line.strip().lower()
    annotated = []
    for word in line.split():
        ngrams = set(dnu.str_to_ngrams(word, dnu.GRAM_SIZE))
        jc_generic = 1.0 * (len(ngrams.intersection(generic_fd)) / 
                            len(ngrams.union(generic_fd)))
        jc_brand = 1.0 * (len(ngrams.intersection(brand_fd)) / 
                          len(ngrams.union(brand_fd)))
        print word, jc_generic, jc_brand
        is_generic = jc_generic > 0.01
        is_brand = jc_brand > 0.01
        if is_generic:
            annotated.append("<GENERIC>%s</GENERIC>" % (word))
        elif is_brand:
            annotated.append("<BRAND>%s</BRAND>" % (word))
        else:
            annotated.append(word)
    print("Input: %s" % (line))
    print("Output: %s" % (" ".join(annotated)))
    i += 1
    if i > 10:
        break
fraw.close()
