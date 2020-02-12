import json
import numpy as np
import os
import sqlite3

from bert_serving.client import BertClient

##################################################################
# Script to generate BERT vectors for all the tweet texts in 
# SQLite3 database.
##################################################################

bc = BertClient()

conn = sqlite3.connect("tweets.db")

fout = open("vectors.tsv", "w")

num_processed = 0
select_sql = """SELECT t_id, t_dttm, t_text FROM tweets"""
cur = conn.cursor()
cur.execute(select_sql)
for row in cur.fetchall():
    if num_processed % 1000 == 0:
        print("{:d} rows processed".format(num_processed))
    try:
        embeddings = bc.encode([row[2]])
    except ValueError:
        continue
    t_vec = ",".join(["{:3e}".format(e) for e in embeddings[0].tolist()])
    fout.write("{:s}\t{:s}\n".format(row[0], t_vec))
    num_processed += 1

print("{:d} rows processed, COMPLETE".format(num_processed))

fout.close()
cur.close()
conn.close()


