import ahocorasick
import joblib
import operator
import os

DATA_DIR = "../../data/entity-graph"
ENTITY_FILES = ["org_syns.csv", "person_syns.csv", "gpe_syns.csv"]

DICT_FILE = os.path.join(DATA_DIR, "entities_dict.pkl")
DICT_KEYS_FILE = os.path.join(DATA_DIR, "entities_dict.keys")

SENTENCES_FILE = os.path.join(DATA_DIR, "sentences.tsv")
COREF_FILE = os.path.join(DATA_DIR, "corefs.tsv")

MATCHED_ENTITIES_FILE = os.path.join(DATA_DIR, "matched_entities.tsv")

def build_automaton():
    print("Building automaton...")
    if os.path.exists(DICT_FILE):
        A = joblib.load(DICT_FILE)
    else:
        fkeys = open(DICT_KEYS_FILE, "w")
        A = ahocorasick.Automaton()
        for entity_file in ENTITY_FILES:
            entity_type = entity_file.split('_')[0][0:3]
            entity_id = 1
            fent = open(os.path.join(DATA_DIR, entity_file), "r")
            for line in fent:
                line = line.strip()
                # print("line:", line)
                if line.startswith("ent_text_x,synonyms"):
                    continue
                display_name, synonyms = line.split(',', 1)
                # print("display_name:", display_name)
                # print("synonyms:", synonyms)
                if len(synonyms) == 0:
                    syn_list = []
                else:
                    syn_list = synonyms.split('|')
                # print("syn_list:", syn_list)
                syn_list.append(display_name)
                unique_syns = list(set(syn_list))
                key = "{:s}{:05d}".format(entity_type[0:3], entity_id)
                fkeys.write("{:s}\t{:s}\n".format(key, display_name))
                for syn in unique_syns:
                    print("...", key, syn)
                    A.add_word(syn, (key, syn))
                entity_id += 1
        A.make_automaton()
        fkeys.close()
        joblib.dump(A, DICT_FILE)
    return A


def find_matches(A, sent_text):
    matched_ents = []
    for char_end, (eid, ent_text) in A.iter(sent_text):
        char_start = char_end - len(ent_text)
        matched_ents.append((eid, ent_text, char_start, char_end))
    # remove shorter subsumed matches
    longest_matched_ents = []
    for matched_ent in sorted(matched_ents, key=lambda x: len(x[1]), reverse=True):
        # print("matched_ent:", matched_ent)
        longest_match_exists = False
        char_start, char_end = matched_ent[2], matched_ent[3]
        for _, _, ref_start, ref_end in longest_matched_ents:
            # print("ref_start:", ref_start, "ref_end:", ref_end)
            if ref_start <= char_start and ref_end >= char_end:
                longest_match_exists = True
                break
        if not longest_match_exists:
            # print("adding match to longest")
            longest_matched_ents.append(matched_ent)
    return longest_matched_ents


def find_corefs(coref_file, sid):
    corefs = []
    fcoref = open(coref_file, "r")
    for line in fcoref:
        if line.startswith("sid"):
            continue
        line = line.strip()
        m_sid, m_start, m_end, m_text, m_main = line.split('\t')
        m_sid = int(m_sid)
        if m_sid == sid:
            corefs.append((int(m_start), int(m_end), m_text, m_main))
        if m_sid > sid:
            break
    fcoref.close()
    return sorted(corefs, key=operator.itemgetter(0), reverse=True)


def replace_corefs(sent_text, corefs):
    sent_out = sent_text
    for start, end, m_text, m_main in corefs:
        sent_out = sent_out[0:start] + m_main + sent_out[end:]
    return sent_out


num_sents, num_ents = 0, 0
A = build_automaton()

print("Finding entities...")
fents = open(MATCHED_ENTITIES_FILE, "w")
fsent = open(SENTENCES_FILE, "r")
for line in fsent:
    if num_sents % 100 == 0:
        print("... {:d} sentences read, {:d} entities written"
            .format(num_sents, num_ents))
    line = line.strip()
    pid, sid, sent_text = line.split('\t')
    # extract and replace coreferences with main text in sentence
    sent_corefs = find_corefs(COREF_FILE, int(sid))
    sent_text = replace_corefs(sent_text, sent_corefs)
    # find matches in the coref enhanced sentences
    matched_ents = find_matches(A, sent_text)
    for eid, ent_text, char_start, char_end in matched_ents:
        fents.write("{:s}\t{:s}\t{:s}\t{:s}\t{:d}\t{:d}\n"
            .format(pid, sid, eid, ent_text, char_start, char_end))
        num_ents += 1
    num_sents += 1

print("... {:d} sentences read, {:d} entities written, COMPLETE"
    .format(num_sents, num_ents))

fsent.close()
fents.close()
