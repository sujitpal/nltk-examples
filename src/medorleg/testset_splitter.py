import numpy as np

def main():
  filelines = {
    "data/sentences/medical.txt": 950887,
    "data/sentences/legal.txt": 837393
  }
  for file, lines in filelines.items():
    test_ids = sorted([int(lines * x) for x in np.random.random((1000,))])
    fn = file.split(".")[0]
    input_file = open(file, 'rb')
    train_file = open(fn + "_train.txt", 'wb')
    test_file = open(fn + "_test.txt", 'wb')
    curr_line = 0
    curr_pos = 0
    for line in input_file:
      if curr_pos < 1000 and curr_line == test_ids[curr_pos]:
        test_file.write(line)
        curr_pos += 1
      else:
        train_file.write(line)
      curr_line += 1
    input_file.close()
    train_file.close()
    test_file.close()
    
if __name__ == "__main__":
  main()
