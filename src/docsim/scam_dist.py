from __future__ import division
import numpy as np
import scipy.sparse as ss

def _s_pos_or_zero(x):
  return x if x > 0 else 0

def _s_zero_mask(x, y):
  return 0 if y == 0 else x

def _s_safe_divide(x, y):
  return 0 if x == 0 or y == 0 else x / y

_v_pos_or_zero = np.vectorize(_s_pos_or_zero)
_v_zero_mask = np.vectorize(_s_zero_mask)
_v_safe_divide = np.vectorize(_s_safe_divide)

def _assymetric_subset_measure(doc1, doc2):
  epsilon = np.ones(doc1.shape) * 2
  filtered = _v_pos_or_zero(epsilon - (_v_safe_divide(doc1, doc2) +
    _v_safe_divide(doc2, doc1)))
  zdoc1 = _v_zero_mask(doc1, filtered)
  zdoc2 = _v_zero_mask(doc2, filtered)
  return np.sum(np.dot(zdoc1, zdoc2)) / np.sum(np.dot(doc1, doc2))

def scam_distance(doc1, doc2):
  asm12 = _assymetric_subset_measure(doc1, doc2)
  asm21 = _assymetric_subset_measure(doc2, doc1)
  return max(asm12, asm21)


