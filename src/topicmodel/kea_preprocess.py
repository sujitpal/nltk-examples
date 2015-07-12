import json
import os
import random
import shutil

JSONS_DIR = "/Users/palsujit/Projects/med_data/mtcrawler/jsons"
KEA_TRAIN_DIR = "/Users/palsujit/Projects/med_data/mtcrawler/kea/train"
KEA_TEST_DIR = "/Users/palsujit/Projects/med_data/mtcrawler/kea/test"

shutil.rmtree(KEA_TRAIN_DIR)
shutil.rmtree(KEA_TEST_DIR)
os.mkdir(KEA_TRAIN_DIR)
os.mkdir(KEA_TEST_DIR)
os.mkdir(os.path.join(KEA_TEST_DIR, "keys"))

for filename in os.listdir(JSONS_DIR):
    print "Converting %s..." % (filename)
    fjson = open(os.path.join(JSONS_DIR, filename), 'rb')
    data = json.load(fjson)
    fjson.close()
    basename = os.path.splitext(filename)[0]
    # do a 30/70 split for training vs test
    train = random.uniform(0, 1) <= 0.1
    txtdir = KEA_TRAIN_DIR if train else KEA_TEST_DIR
    ftxt = open(os.path.join(txtdir, basename + ".txt"), 'wb')
    ftxt.write(data["text"].encode("utf-8"))
    ftxt.close()
    # write keywords
    keydir = KEA_TRAIN_DIR if train else os.path.join(KEA_TEST_DIR, "keys")    
    fkey = open(os.path.join(keydir, basename + ".key"), 'wb')     
    keywords = data["keywords"]
    for keyword in keywords:
        fkey.write("%s\n" % (keyword.encode("utf-8")))
    fkey.close()

