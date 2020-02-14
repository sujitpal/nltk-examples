import operator
import os
import spacy
import neuralcoref
import nltk

PRONOUNS = set(["he", "she", "him", "her", "they", "their", "it"])

DATA_DIR = "../../data/entity-graph"

TEXT_FILENAME = os.path.join(DATA_DIR, "db-article.txt")
SENTENCES_FILENAME = os.path.join(DATA_DIR, "sentences.tsv")
COREF_FILENAME = os.path.join(DATA_DIR, "corefs.tsv")

def get_coref_clusters(ptext, nlp, offset=0):
    output_clusters = []
    doc = nlp(ptext)
    for coref_cluster in doc._.coref_clusters:
        main_text = coref_cluster.main.text
        for mention in coref_cluster.mentions:
            if nltk.edit_distance(main_text, mention.text) <= 5:
                continue
            if mention.start_char < offset:
                # mentions from previous paragraph, don't report
                continue
            output_clusters.append((mention.start_char - offset, 
                                    mention.end_char - offset, 
                                    mention.text,
                                    main_text))

    return output_clusters


def partition_mentions_by_sentence(mentions, ptext, para_id, nlp):
    curr_sid = 0
    fsent = open(SENTENCES_FILENAME, "r")
    for line in fsent:
        pid, sid, sent = line.strip().split('\t')
        pid, sid = int(pid), int(sid)
        if pid == para_id:
            curr_sid = sid
            break
    fsent.close()
    partitioned_mentions = []
    sent_bounds = [(sid, s.start_char, s.end_char) for sid, s in enumerate(nlp(ptext).sents)]
    for mention in mentions:
        m_sid = None
        m_start, m_end, m_text, m_main = mention
        for sent_bound in sent_bounds:
            sid, s_start, s_end = sent_bound
            if m_start >= s_start and m_end <= s_end:
                m_sid = sid
                m_start -= s_start
                m_end -= s_start
                break
        if m_sid is not None:
            partitioned_mentions.append((curr_sid + m_sid, m_start, m_end, m_text, m_main))
    return partitioned_mentions


nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)

fcoref = open(COREF_FILENAME, "w")
ftext = open(TEXT_FILENAME, "r")

fcoref.write("sid\tm_start\tm_end\tm_text\tm_main\n")

prev_ptext = None
curr_pid, curr_sid = 0, 0
for ptext in ftext:
    if curr_pid % 100 == 0:
        print("{:d} paragraphs processed".format(curr_pid))
    ptext = ptext.strip()
    # skip empty lines
    if len(ptext) == 0:
        continue
    # does the text have pronouns
    doc = nlp(ptext)
    tokens = set([token.text.lower() for token in doc])
    if len(tokens.intersection(PRONOUNS)) == 0:
        curr_pid += 1
        continue
    output_clusters = get_coref_clusters(ptext, nlp)
    # if we couldn't find corefs even though we had pronouns lets
    # increase the scope to previous paragraph as well
    if len(output_clusters) == 0 and prev_ptext is not None:
        output_clusters = get_coref_clusters(" ".join([prev_ptext, ptext]), 
            nlp, offset=len(prev_ptext)+1)

    # partition the list among individual sentences
    partitioned_mentions = partition_mentions_by_sentence(
        output_clusters, ptext, curr_pid, nlp)
    for mention_p in partitioned_mentions:
        pm_sid, pm_start, pm_end, pm_text, pm_main = mention_p
        fcoref.write("{:d}\t{:d}\t{:d}\t{:s}\t{:s}\n".format(
            pm_sid, pm_start, pm_end, pm_text, pm_main))

    # set previous paragraph (in case needed, see above)
    prev_ptext = ptext
    curr_pid += 1

print("{:d} paragraphs processed, COMPLETE".format(curr_pid))

ftext.close()
fcoref.close()
