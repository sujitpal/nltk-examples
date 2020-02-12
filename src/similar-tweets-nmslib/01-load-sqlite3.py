import os
import sqlite3

##################################################################
# Script to read and parse multiple tweet files and load them 
# into a SQLite3 DB for later retrieval.
##################################################################

DATA_DIR = "../data"
INPUT_DIR = os.path.join(DATA_DIR, "Health-Tweets")
DB_FILE = os.path.join(DATA_DIR, "tweets.db")

# create database
conn = sqlite3.connect(DB_FILE)

# create table if not exists
cur = conn.cursor()
try:
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tweets (
        t_id VARCHAR(32) NOT NULL,
        t_dttm VARCHAR(50) NOT NULL,
        t_text VARCHAR(255) NOT NULL
    )
    """)
    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS ix_tweets ON tweets(t_id)
    """)
except sqlite3.Error as e:
    print("Failed to create table tweets and unique index")
    raise e
finally:
    if cur: cur.close()


num_written = 0
insert_sql = """
    INSERT INTO tweets(t_id, t_dttm, t_text) VALUES (?, ?, ?)
"""
for filename in os.listdir(INPUT_DIR):
    print("Now processing: {:s}".format(filename))
    fin = open(os.path.join(INPUT_DIR, filename), "r", encoding="utf8")
    for line in fin:
        cols = line.strip().split('|')
        if len(cols) != 3:
            continue
        if num_written % 1000 == 0:
            print("{:d} rows added".format(num_written))
            conn.commit()
        t_id, t_dttm, t_text = cols
        t_text = " ".join([w for w in t_text.split() if not w.startswith("http://")])
        # print(t_id, t_dttm, t_text)
        try:
            cur = conn.cursor()
            cur.execute(insert_sql, (t_id, t_dttm, t_text))
        except sqlite3.Error as e:
            print("Error inserting data")
            raise e
        finally:
            if cur: cur.close()
        num_written += 1
    fin.close()

print("{:d} rows added, COMPLETE".format(num_written))
conn.commit()
