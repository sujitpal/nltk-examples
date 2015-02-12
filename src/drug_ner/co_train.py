# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import drug_ner_utils as dnu
import matplotlib.pyplot as plt
import numpy as np
import os

MAX_ITERS = 10
EST_POSITIVE = 0.7
MAX_FEATURES = 3000

def conservative_min(xs):
    # remove outliers
    q25, q75 = np.percentile(xs, [25, 75])
    iqr = q75 - q25
    lb = q25 - (1.5 * iqr)
    ub = q75 + (1.5 * iqr)
    xs_con = xs[(xs >= lb) & (xs <= ub)]
    return np.min(xs_con)
    
    
for borg in ["generic", "brand"]:
    X, y, vec = dnu.vectorize("unlabeled.txt", "%s_positive.txt" % (borg), 
                              MAX_FEATURES)

    y_pos = y[y == 1]
    num_positives = [y_pos.shape[0]]

    clf = LinearSVC()
    clf.fit(X, y)

    num_iters = 0
    while (num_iters < MAX_ITERS):
        print("Iteration #%d, #-positive examples: %d" % 
              (num_iters, num_positives[-1]))
        confidence = clf.decision_function(X)
        min_pos_confidence = conservative_min(confidence[y_pos])
        y_pos = np.where(confidence >= min_pos_confidence)[0]
#        if y_pos.shape[0] <= num_positives[-1]:
#            break
        num_positives.append(y_pos.shape[0])
        y = np.zeros(y.shape)
        y[y_pos] = 1
        clf = LinearSVC()
        clf.fit(X, y)
        joblib.dump(y, os.path.join(dnu.DATA_DIR, "y_%s_%d.pkl" % 
                    (borg, num_iters)))
        num_iters += 1
    
    # visualize output
    plt.plot(np.arange(len(num_positives)), num_positives, "b-")
    plt.plot(np.arange(len(num_positives)), 
             X.shape[0] * EST_POSITIVE * np.ones(len(num_positives)), 'r--')
    plt.title("Cotraining for %s classifier (%d features)" % 
              (borg.title(), MAX_FEATURES))
    plt.xlabel("Iterations")
    plt.ylabel("#-Positives")
    plt.show()
    
