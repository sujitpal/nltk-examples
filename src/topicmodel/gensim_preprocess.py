import json
import os

JSONS_DIR = "/home/sujit/Projects/mlia-examples/data/mtcrawler/jsons"
TEXTS_DIR = "/home/sujit/Projects/mlia-examples/data/mtcrawler/gensim"

for fn in os.listdir(JSONS_DIR):
    print "Converting JSON: %s" % (fn)
    fjson = open(os.path.join(JSONS_DIR, fn), 'rb')
    data = json.load(fjson)
    fjson.close()
    tfn = os.path.splitext(fn)[0] + ".txt"
    ftext = open(os.path.join(TEXTS_DIR, tfn), 'wb')
    ftext.write(data["text"].encode("utf-8"))
    ftext.close()
