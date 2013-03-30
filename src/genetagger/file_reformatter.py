# Reformats supplied input file to form parseable by NLTK corpus readers.
def reformat(file_in, file_out, is_tagged):
  fin = open(file_in, 'rb')
  fout = open(file_out, 'wb')
  sent = []
  for line in fin:
    line = line.strip()
    if len(line) == 0:
      if is_tagged:
        fout.write(" ".join(["/".join([word, tag]) for word, tag in sent]) + "\n")
      else:
        fout.write(" ".join([word for word in sent]) + "\n")
      sent = []
      continue
    if is_tagged:
      word, tag = line.split(" ")
      sent.append((word, tag))
    else:
      sent.append(line)
  fin.close()
  fout.close()

def main():
  reformat("gene.train", "gene.train.blog", True)
  reformat("gene.key", "gene.validate.blog", True)
  reformat("gene.test", "gene.test.blog", False)
  
if __name__ == "__main__":
  main()
