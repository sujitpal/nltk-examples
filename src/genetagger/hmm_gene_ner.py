from __future__ import division
import nltk
import nltk.probability

class Accumulator:
  """
  Convenience class to accumulate all the frequencies
  into a set of data structures.
  """
  def __init__(self):
    self.words = set()
    self.tags = set()
    self.priorsFD = nltk.FreqDist()
    self.transitionsCFD = nltk.ConditionalFreqDist()
    self.outputsCFD = nltk.ConditionalFreqDist()
    self.prevTag = None

  def add(self, word, tag):
    self.words.add(word)
    self.tags.add(tag)
    self.priorsFD.inc(tag)
    self.outputsCFD[tag].inc(word)
    if self.prevTag is not None:
      self.transitionsCFD[self.prevTag].inc(tag)
    self.prevTag = tag

def train(train_file):
  """
  Read the file and populate the various frequency and
  conditional frequency distributions and build the HMM
  off these data structures.
  """
  f = open(train_file, 'rb')
  acc = Accumulator()
  for line in f:
    line = line.strip()
    if line is None:
      # end of file
      acc.add("<STOP>", "STOP")
      break
    elif len(line) == 0:
      # end of sentence
      acc.add("<STOP>", "STOP")
      acc.add("<START-0>", "START-0")
      acc.add("<START-1>", "START-1")
      continue
    else:
      word, tag = line.split(" ")
      acc.add(word, tag)
  f.close()
  return nltk.HiddenMarkovModelTagger(list(acc.words), list(acc.tags),
    nltk.ConditionalProbDist(acc.transitionsCFD, nltk.ELEProbDist),
    nltk.ConditionalProbDist(acc.outputsCFD, nltk.ELEProbDist),
    nltk.ELEProbDist(acc.priorsFD))

def calc_accuracy(predicted, actual):
  """
  Returns the number of cases where prediction and actual NER
  tags are the same, divided by the number of tags for the
  sentence.
  """
  return (len(filter(lambda x: x[0] == x[1], zip(predicted, actual))) /
    len(predicted))

def validate(hmm, validation_file):
  """
  Tests the HMM against the validation file.
  """
  tot_acc = 0
  nsents = 0
  f = open(validation_file, 'rb')
  words = ["<START-0>", "<START-1>"]
  tags = ["START-0", "START-1"]
  for line in f:
    line = line.strip()
    if line is None:
      # end of file, exit
      words.append("<STOP>")
      tags.append("STOP")
      break
    elif len(line) == 0:
      words.append("<STOP>")
      tags.append("STOP")
      predicted_tags = hmm.best_path(words)
      tot_acc += calc_accuracy(tags, predicted_tags)
      nsents += 1
      words = ["<START-0>", "<START-1>"]
      tags = ["START-0", "START-1"]
    else:
      word, tag = line.split(" ")
      words.append(word)
      tags.append(tag)
  f.close()
  tot_acc += calc_accuracy(tags, predicted_tags)
  nsents += 1
  print "Validation Accuracy=", (tot_acc / nsents)

def write_result(fout, hmm, words):
  """
  Writes out the result in the required format.
  """
  tags = hmm.best_path(words)
  for word, tag in zip(words, tags)[2:-1]:
    fout.write("%s %s\n" % (word, tag))
  fout.write("\n")
  
def test(hmm, test_file, result_file):
  """
  Tests the HMM against the test file (without tags) and writes
  out the results to the result file.
  """
  fin = open(test_file, 'rb')
  fout = open(result_file, 'wb')
  words = ["<START-0>", "<START-1>"]
  for line in fin:
    line = line.strip()
    if line is None:
      # end of file, exit
      words.append("<STOP>")
      break
    elif len(line) == 0:
      words.append("<STOP>")
      write_result(fout, hmm, words)
      words = ["<START-0>", "<START-1>"]
    else:
      words.append(line)
  write_result(fout, hmm, words)
  fin.close()
  fout.close()

def main():
  hmm = train("gene.train")
  validate(hmm, "gene.key")
  test(hmm, "gene.dev", "gene.dev.out")

if __name__ == "__main__":
  main()