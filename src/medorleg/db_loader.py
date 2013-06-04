import sqlite3 as sql

def is_empty(conn):
  cur = conn.cursor()
  cur.execute("select name from sqlite_master where type='table'")
  rows = cur.fetchall()
  return len(rows) == 0

def create_tables(conn):
  if not is_empty(conn):
    return
  cur = conn.cursor()
  cur.executescript("""
    create table m3 (w1 text, w2 text, w3 text, freq integer);
    create table m2 (w2 text, w3 text, freq integer);
    create table m1 (w3 text, freq integer);
    create table l3 (w1 text, w2 text, w3 text, freq integer);
    create table l2 (w2 text, w3 text, freq integer);
    create table l1 (w3 text, freq integer);
  """)
  conn.commit()

def gram_to_list(gram):
  return [x[1:-1] for x in gram[1:-1].split(", ")]

def populate_tables(conn, infn, t3n, t2n, t1n):
  cur = conn.cursor()
  infile = open(infn, 'rb')
  i = 0
  for line in infile:
    if i % 1000 == 0:
      print "Processing %s, line: %d" % (infn, i)
    gram, count = line.strip().split("\t")
    gramlist = gram_to_list(gram)
    if len(gramlist) == 3:
      cur.execute("insert into %s(w1,w2,w3,freq)values(?,?,?,?)" % (t3n),
        (gramlist[0], gramlist[1], gramlist[2], int(count)))
    elif len(gramlist) == 2:
      cur.execute("insert into %s(w2,w3,freq)values(?,?,?)" % (t2n),
        (gramlist[0], gramlist[1], int(count)))
    else:
      cur.execute("insert into %s(w3,freq)values(?,?)" % (t1n),
        (gramlist[0], int(count)))
    i += 1
  infile.close()
  conn.commit()

def build_indexes(conn):
  print "Building indexes..."
  cur = conn.cursor()
  cur.executescript("""
    create unique index ix_m3 on m3(w1,w2,w3);
    create unique index ix_m2 on m2(w2,w3);
    create unique index ix_m1 on m1(w3);
    create unique index ix_l3 on l3(w1,w2,w3);
    create unique index ix_l2 on l2(w2,w3);
    create unique index ix_l1 on l1(w3);
  """)
  conn.commit()
  
def main():
  conn = sql.connect("data/db/ngram_freqs.db")
  create_tables(conn)
  populate_tables(conn, "data/counts/medical_count.txt", "m3", "m2", "m1")
  populate_tables(conn, "data/counts/legal_count.txt", "l3", "l2", "l1")
  build_indexes(conn)
  conn.close()

if __name__ == "__main__":
  main()