import argparse
import operator
import os

DATA_DIR = "../../data/entity-graph"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input file")
parser.add_argument("-o", "--output", required=True, help="Output file")
args = parser.parse_args()

input_file = args.input
output_file = args.output

fout = open(os.path.join(DATA_DIR, output_file), "w")
fin = open(os.path.join(DATA_DIR, input_file), "r")
for line in fin:
    line = line.strip()
    display_name, synonyms = line.split(',', 1)
    synonym_list = synonyms.split('|')
    synonym_list.append(display_name)
    unique_synonyms = sorted(list(set(synonym_list)), key=len, reverse=True)
    display_name = unique_synonyms[0]
    synonyms = '|'.join(unique_synonyms[1:])
    fout.write("{:s},{:s}\n".format(display_name, synonyms))

fin.close()
fout.close()
