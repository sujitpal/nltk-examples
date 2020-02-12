import nmslib
import numpy as np
import os
import sqlite3
import time

DATA_DIR = "../data"

TWEET_DB = os.path.join(DATA_DIR, "tweets.db")
VECTORS_FILE = os.path.join(DATA_DIR, "vectors.tsv")
RESULTS_FILE = os.path.join(DATA_DIR, "results.tsv")
NMS_INDEX = os.path.join(DATA_DIR, "tweet-vectors.index")

MAX_NUM_VECTORS = 63111
INDEX_SAMPLES = 63111
QUERY_SAMPLES = 50


def lookup_tweet_by_id(tweet_id):
    try:
        conn = sqlite3.connect(TWEET_DB)
        cur = conn.cursor()
        cur.execute("""SELECT t_text FROM tweets WHERE t_id = '%s' """ % (tweet_id))
        row = cur.fetchone()
        return row[0]
    except sqlite3.Error as e:
        raise e
    finally:
        if cur: cur.close()
        if conn: conn.close()


# build vector data for required number of samples
index_positions = set(
    np.random.random_integers(low=0, high=MAX_NUM_VECTORS, size=INDEX_SAMPLES)
    .tolist())
query_positions = set(
    np.random.random_integers(low=0, high=MAX_NUM_VECTORS, size=QUERY_SAMPLES)
    .tolist())

index_vecs = np.empty((INDEX_SAMPLES, 768))
query_vecs = np.empty((QUERY_SAMPLES, 768))
index_pos2id, query_pos2id = {}, {}

fvec = open(VECTORS_FILE, "r")
curr_index_position, curr_query_position = 0, 0
for lid, line in enumerate(fvec):
    if lid in index_positions or lid in query_positions:
        t_id, t_vec = line.strip().split('\t')
        t_vec_arr = np.array([float(v) for v in t_vec.split(',')])
        if lid in index_positions:
            index_vecs[curr_index_position] = t_vec_arr
            index_pos2id[curr_index_position] = t_id
            curr_index_position += 1
        else: # lid in query_positions:
            query_vecs[curr_query_position] = t_vec_arr
            query_pos2id[curr_query_position] = t_id            
            curr_query_position += 1
    else:
        continue

fvec.close()

# load
start_tm = time.time()
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(index_vecs)
index.createIndex({'post': 2}, print_progress=True)
elapsed_tm = time.time() - start_tm
print("load elapsed time (s): {:.3f}".format(elapsed_tm))

index.saveIndex(NMS_INDEX, save_data=True)

fout = open(RESULTS_FILE, "w")
query_times = []
for i in range(query_vecs.shape[0]):
    try:
        start_tm = time.time()
        q_tid = query_pos2id[i]
        q_text = lookup_tweet_by_id(q_tid)
        fout.write("query: {:s} ({:s})\n".format(q_text, q_tid))
        rids, distances = index.knnQuery(query_vecs[i], k=10)
        for rid, distance in zip(rids, distances):
            r_tid = index_pos2id[rid]
            r_text = lookup_tweet_by_id(r_tid)
            fout.write("{:.3f} {:s} {:s}\n".format(distance, r_tid, r_text))
        query_times.append(time.time() - start_tm)
    except KeyError:
        continue

fout.close()
print("average query elapsed time (s): {:.3f}".format(sum(query_times) / len(query_times)))