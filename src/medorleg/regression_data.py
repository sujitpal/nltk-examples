from __future__ import division

import math
import sqlite3 as sql

def get_base_counts(conn, morl):
  cur = conn.cursor()
  cur.execute("select count(*), sum(freq) from %s1" % (morl))
  v, n = cur.fetchall()[0]
  return n, v

def gram_to_list(gram):
  return [x[1:-1] for x in gram[1:-1].split(", ")]

def build_regdata(conn, morl, infn, outX, outY):
  cur = conn.cursor()
  infile = open(infn, 'rb')
  xfile = open(outX, 'wb')
  yfile = open(outY, 'wb')
  n, v = get_base_counts(conn, morl)
  i = 0
  for line in infile:
    gram, freq = line.strip().split("\t")
    gramlist = gram_to_list(gram)
    if len(gramlist) == 3:
      cur.execute("select freq from %s3 where w1 = ? and w2 = ? and w3 = ?"
        % (morl), gramlist)
      rows = cur.fetchall()
      freq3 = 0 if len(rows) == 0 else rows[0][0]
      cur.execute("select freq from %s2 where w2 = ? and w3 = ?" %
        (morl), gramlist[1:])
      rows = cur.fetchall()
      freq2 = 0 if len(rows) == 0 else rows[0][0]
      cur.execute("select freq from %s1 where w3 = ?" % (morl), gramlist[2:])
      rows = cur.fetchall()
      freq1 = 0 if len(rows) == 0 else rows[0][0]
      y = math.log(freq3) - math.log(n)
      x0 = math.log(freq3) - math.log(freq2)
      x1 = math.log(freq2) - math.log(freq1)
      x2 = math.log(freq1 + 1) / math.log(n + v)
      print morl, x0, x1, x2, y
      xfile.write("%s %s %s\n" % (x0, x1, x2))
      yfile.write("%s\n" % (y))
  infile.close()
  xfile.close()
  yfile.close()

def main():
  conn = sql.connect("data/db/ngram_freqs.db")
  build_regdata(conn, "m", "data/counts/medical_count.txt",
    "data/regdata/medical_X.txt", "data/regdata/medical_y.txt")
  build_regdata(conn, "l", "data/counts/legal_count.txt",
    "data/regdata/legal_X.txt", "data/regdata/legal_y.txt")
  conn.close()
  
if __name__ == "__main__":
  main()
