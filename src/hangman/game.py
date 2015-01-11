# -*- coding: utf-8 -*-
import operator
from random import random

MAX_GUESSES_PER_GAME = 11
MAX_WORDS_IN_SUBSET = 25

########################### Preprocessing ############################

non_ascii_mappings = list([
    ('á', 'a'), ('å', 'a'), ('ä', 'a'), ('â', 'a'), ('Å', 'a'), ('Ã', 'a'),
    ('ç', 'c'), ('¢', 'c'),
    ('é', 'e'), ('é', 'e'), ('è', 'e'), ('ê', 'e'),
    ('í', 'i'),
    ('ñ', 'n'),
    ('ó', 'o'), ('ó', 'o'), ('ö', 'o'), ('ô', 'o'),
    ('ü', 'u'), ('û', 'u'),
    ('´', '\''), ('»', '"')
])

def ascii_fold(s):
    for x in non_ascii_mappings:
        s = s.replace(x[0], x[1])
    return s

def preprocess(dictfile):    
    fwords = open(dictfile, 'rb')
    wset = set()
    for line in fwords:
        word = line.strip().lower()
        word = ascii_fold(word)
        if word.endswith("'s"):
            word = word[:-2]
        word = word.replace(" ", "").replace("'", "").replace("\"", "")
        wset.add(word)
    fwords.close()
    return list(wset)

############################# Proposer Side #############################
    
def select_secret_word(words):
    widx = int(random() * len(words))
    return words[widx]
    
def find_all_match_positions(secret_word, guess_char):
    positions = []
    curr_pos = 0
    while curr_pos < len(secret_word):
        curr_pos = secret_word.find(guess_char, curr_pos)
        if curr_pos < 0:
            break
        positions.append(curr_pos)
        curr_pos += 1
    return positions    
    
def update_guessed_word(guessed_word, matched_positions, guessed_char):
    for pos in matched_positions:
        guessed_word[pos] = guessed_char
    
def is_solved(guessed_word):
    chars_remaining = len(filter(lambda x: x == '_', guessed_word))
    return chars_remaining == 0
    

############################# Solver Side ###############################

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def most_frequent_char(words, previously_guessed):
    cmap = {x:0 for x in letters}
    for word in words:
        cset = set()
        for c in word:
            if c not in previously_guessed:
                cset.add(c)
        for c in cset:
            cmap[c] += 1
    return sorted(map(lambda x: (x, cmap[x]), cmap.keys()), 
                 key=operator.itemgetter(1), reverse=True)[0][0]
    
def best_guess(words, word_len, bad_guesses, good_guesses, guessed_word):
    temp_words = filter(lambda x: len(x) == word_len, words)
    if len(bad_guesses) > 0:
        for bad_guess in bad_guesses:
            temp_words = filter(lambda x: x.find(bad_guess) == -1, temp_words)
    if len(good_guesses) > 0:
        for good_guess in good_guesses:
            temp_words = filter(lambda x: x.find(good_guess) > -1, temp_words)
    previously_guessed = set(filter(lambda x: x != '_', guessed_word))
    return temp_words, most_frequent_char(temp_words, previously_guessed)
    
def init_guess(wordlen):
    initial_guess = []
    for i in range(wordlen):
        initial_guess.append("_")
    return initial_guess

def match_words_against_template(words, guessed_word):
    if len(words) > MAX_WORDS_IN_SUBSET:
        return words
    matched_words = []
    for word in words:
        word_chars = [c for c in word]
        merged = zip(guessed_word, word_chars)
        diff = len(filter(lambda x: x[0] != x[1], 
                          filter(lambda x: x[0] != '_', merged)))
        if diff == 0:
            matched_words.append(word)
        if len(matched_words) > 1:
            break
    return matched_words

def replace_guessed_word(guessed_word, matched_word):
    matched_chars = [c for c in matched_word]
    for i in range(len(matched_chars)):
        guessed_word[i] = matched_chars[i]
    
################################# Game ###################################
    
def single_round(words, debug=False):
    solver_wins = False
    secret_word = select_secret_word(words)
    if debug:
        print "secret word:", secret_word
    word_len = len(secret_word)
    bad_guesses = set()
    good_guesses = set()
    guessed_word = init_guess(word_len)
    for num_guesses in range(MAX_GUESSES_PER_GAME):
        filtered_words, guess_char = best_guess(words, word_len, bad_guesses, 
                                                good_guesses, guessed_word)
        if debug:
            print "guessed char:", guess_char
        matched_positions = find_all_match_positions(secret_word, guess_char)
        if len(matched_positions) == 0:
            bad_guesses.add(guess_char)
        else:
            good_guesses.add(guess_char)
        update_guessed_word(guessed_word, matched_positions, guess_char)
        matched_words = match_words_against_template(filtered_words, guessed_word)
        if len(matched_words) == 1:
            replace_guessed_word(guessed_word, matched_words[0])
        if debug:
            print "#", num_guesses, "guess:", " ".join(guessed_word)
        if is_solved(guessed_word):
            solver_wins = True
            break
    return len(secret_word), solver_wins, num_guesses
    
def multiple_rounds(words, num_games, report_file):
    fdata = open(report_file, 'wb')
    for i in range(num_games):
        word_len, solver_wins, num_guesses = single_round(words, False)
        fdata.write("%d,%d,%d\n" % (word_len, 1 if solver_wins else 0, num_guesses))
    fdata.close()

################################# Main ###################################

words = preprocess("/usr/share/dict/words")

#single_round(words, True)

multiple_rounds(words, 10000, "hangman.csv")
