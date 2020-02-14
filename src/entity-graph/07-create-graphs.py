import itertools
import os
import pandas as pd

DATA_DIR = "../../data/entity-graph"

DICT_KEYS_FILE = os.path.join(DATA_DIR, "entities_dict.keys")
MATCHED_ENTITIES_FILE = os.path.join(DATA_DIR, "matched_entities.tsv")

NODE_FILE = os.path.join(DATA_DIR, "neo4j_nodes.csv")
EDGE_FILE = os.path.join(DATA_DIR, "neo4j_edges.csv")

# generate Nodes CSV
fnodes = open(NODE_FILE, "w")
fnodes.write("eid:ID,ename,:LABEL\n")
fkeys = open(DICT_KEYS_FILE, "r")
for line in fkeys:
    eid, ename = line.strip().split('\t')
    fnodes.write(','.join([eid, ename, eid[0:3].upper()]) + '\n')
fkeys.close()
fnodes.close()

# generate Edges CSV 
ents_df = pd.read_csv(MATCHED_ENTITIES_FILE, sep='\t',
    names=["pid", "sid", "eid", "etext", "estart", "estop"])
edges_df = (
    ents_df[["sid", "eid"]]     # extract (sid, eid)
    .groupby("sid")["eid"]      # group by sid
    .apply(list)                # (sid, list[eid, ...])
    .reset_index(name="eids")
)
# generate entity ID pairs: (sid, list[(eid1, eid2), ...])
edges_df["eids"] = (
    edges_df["eids"]
    .apply(lambda xs: list(set(xs)))
    .apply(lambda xs: [x for x in itertools.combinations(xs, 2)])
)
# unstack the list of pairs
rows = []
for row in edges_df.itertuples():
    # note: 1 based because Index is 0
    sid = row[1]
    for edge in row[2]:
        rows.append([edge[0], edge[1], sid])
edges_df = pd.DataFrame(data=rows, columns=[":START_ID", ":END_ID", "sid"])
edges_df[":TYPE"] = "REL"
# print(edges_df.head())
edges_df.to_csv(EDGE_FILE, index=False)

#############################################################################
# Load these files into neo4j by doing the following:
#   1. cd $NEO4J_HOME/data/databases
#   2. rm -rf *
#   3. cd $NEO4J_HOME
#   4. bin/neo4j-admin import --nodes=/path/to/neo4j_nodes.csv \
#           --relationships=/path/to/neo4j_edges.csv 
#   5. bin/neo4j start
#############################################################################
