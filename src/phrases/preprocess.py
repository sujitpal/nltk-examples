from __future__ import division

import json
import sys

def main():
  if len(sys.argv) != 2:
    print "Usage: %s /path/to/twitter/json/list.txt"
    sys.exit(-1)
  fin = open(sys.argv[1], 'rb')
  fout = open("twitter_messages.txt", 'wb')
  for line in fin:
    try:
      data = json.loads(line.strip())
      lang = data["lang"]
      if lang == "en":
        tweet = data["text"]
        tweet = tweet.replace("\n", " ").replace("\\s+", " ")
        tweet = tweet.encode("ascii", "ignore")
        if len(tweet) == 0:
          continue
        fout.write("%s\n" % (tweet))
    except KeyError:
      continue
  fin.close()
  fout.close()
      

if __name__ == "__main__":
  main()