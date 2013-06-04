from __future__ import division

import math
import sqlite3 as sql

import cPickle as pickle
import nltk
import numpy as np
import string

def normalize_numeric(x):
  xc = x.translate(string.maketrans("", ""), string.punctuation)
  return "_NNN_" if xc.isdigit() else x

def normalize_stopword(x, stopwords):
  return "_SSS_" if str(x) in stopwords else x

def get_trigrams(sentence, stopwords, porter):
  words = nltk.word_tokenize(sentence)
  words = [word.lower() for word in words]
  words = [normalize_numeric(word) for word in words]
  words = [normalize_stopword(word, stopwords) for word in words]
  words = [porter.stem(word) for word in words]
  return nltk.trigrams(words)

def get_base_counts(conn, morl):
  cur = conn.cursor()
  cur.execute("select count(*), sum(freq) from %s1" % (morl))
  v, n = cur.fetchall()[0]
  return n, v

def load_model_coeffs(model):
  norm = np.sum(model.coef_)
  return np.array([model.coef_[0] / norm,
    model.coef_[1] / norm,
    model.coef_[2] / norm,
    model.intercept_ / norm])

def calc_prob(trigrams, conn, coeffs, morl, n, v):
  joint_log_prob = 0.0
  cur = conn.cursor()
  for trigram in trigrams:
    cur.execute("select freq from %s3 where w1 = ? and w2 = ? and w3 = ?"
      % (morl), trigram)
    rows = cur.fetchall()
    freq3 = 0 if len(rows) == 0 else rows[0][0]
    cur.execute("select freq from %s2 where w2 = ? and w3 = ?" %
      (morl), trigram[1:])
    rows = cur.fetchall()
    freq2 = 0 if len(rows) == 0 else rows[0][0]
    cur.execute("select freq from %s1 where w3 = ?" % (morl), trigram[2:])
    rows = cur.fetchall()
    freq1 = 0 if len(rows) == 0 else rows[0][0]
    freqs = np.array([
      0 if freq3 == 0 else freq3 / freq2,
      0 if freq2 == 0 else freq2 / freq1,
      0 if freq1 == 0 else (freq1 + 1) / (n + v),
      1.0])
    joint_log_prob += math.log(1 + np.dot(coeffs, freqs))
  return joint_log_prob

def eval_model(medmodelfn, legmodelfn, testfn, stopwords, porter, conn):
  pos = {"M": 0, "L": 1}
  stats = np.zeros((2, 2))
  med_params = load_model_coeffs(pickle.load(open(medmodelfn, 'rb')))
  leg_params = load_model_coeffs(pickle.load(open(legmodelfn, 'rb')))
  mn, mv = get_base_counts(conn, "m")
  ln, lv = get_base_counts(conn, "l")
  testfile = open(testfn, 'rb')
  i = 0
  for line in testfile:
    if i % 100 == 0:
      print "Tested %d/1000 test cases..." % (i)
    i += 1
    cols = line.strip().split("|")
    trigrams = get_trigrams(cols[1], stopwords, porter)
    med_prob = calc_prob(trigrams, conn, med_params, "m", mn, mv)
    leg_prob = calc_prob(trigrams, conn, leg_params, "l", ln, lv)
    ytruth = cols[0]
    ypred = "M" if med_prob > leg_prob else "L"
    print "...", i, ytruth, ypred
    stats[pos[ytruth], pos[ypred]] += 1
  return stats

def calc_acc(stats):
  return np.sum(np.diag(stats)) / np.sum(stats)

def main():
  stopwords = nltk.corpus.stopwords.words("english")
  porter = nltk.PorterStemmer()
  conn = sql.connect("data/db/ngram_freqs.db")
  med_stats = eval_model("data/regdata/medical.pkl",
    "data/regdata/legal.pkl", "data/sentences/medical_test.txt",
    stopwords, porter, conn)
  print "confusion matrix (med), acc=", calc_acc(med_stats)
  print med_stats
  leg_stats = eval_model("data/regdata/medical.pkl",
    "data/regdata/legal.pkl", "data/sentences/legal_test.txt",
    stopwords, porter, conn)
  print "confusion matrix (leg), acc=", calc_acc(leg_stats)
  print leg_stats
  merged_stats = med_stats + leg_stats
  print "confusion matrix (merged), acc=", calc_acc(merged_stats)
  print merged_stats
  conn.close()

if __name__ == "__main__":
  main()
