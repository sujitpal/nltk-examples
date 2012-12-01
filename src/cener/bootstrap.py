#!/usr/bin/python

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re

stopwords = set(["The", "This", "Though", "While", 
  "Using", "It", "Its", "A", "An", "As", "Now",
  "At", "But", "Although", "Am", "Perhaps",
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"])

def iotag(token):
  # remove stopwords
  if token in stopwords:
    return False
  if (re.match("^[A-Z].*", token) or
      re.match("^[a-z][A-Z].*", token) or
      re.search("[0-9]", token) or
      token == ",s"):
    return True
  else:
    return False

# if current iotag == "I" and (prev iotag == "I" or next iotag == "I"
# then keep the iotag value else flip it
def modify_tags(pairs):
  output_tags = []
  idx = 0
  for pair in pairs:
    if pair[1]:
      if idx == 0:
        output_tags.append((pair[0], pair[1] and pairs[idx+1][1]))
      elif idx == len(pairs):
        output_tags.append((pair[0], pair[1] and pairs[idx-1][1]))
      else:
        output_tags.append((pair[0], pair[1] and
          (pairs[idx-1][1] or pairs[idx+1][1])))
    else:
      output_tags.append(pair)
    idx = idx + 1
  return output_tags

def partition_pairs(pairs):
  output_pairs_list = []
  output_pairs = []
  for pair in pairs:
    if pair[1]:
      output_pairs.append(pair)
    else:
      if len(output_pairs) > 0:
        output_pairs_list.append(output_pairs)
        output_pairs = []
  return output_pairs_list

def main():
  ce_words = set()
  input = open("cnet_reviews.txt", 'rb')
  for line in input:
    line = line[:-1]
    if len(line.strip()) == 0:
      continue
    sents = sent_tokenize(line)
    for sent in sents:
#      print sent
      tokens = word_tokenize(sent)
      iotags = map(lambda token: iotag(token), tokens)
      ce_pairs_list = partition_pairs(modify_tags(zip(tokens, iotags)))
      if len(ce_pairs_list) == 0:
        continue
      for ce_pairs in ce_pairs_list:
        print " ".join(map(lambda pair: pair[0], ce_pairs))
        for ce_pair in ce_pairs:
          ce_words.add(ce_pair[0])
  input.close()

if __name__ == "__main__":
  main()
