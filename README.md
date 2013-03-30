nltk-examples
=============

src/book/
=========

Worked examples from the NLTK Book

src/cener/
===========
A Consumer Electronics Named Entity Recognizer - uses an NLTK Maximum Entropy
Classifier and IOB tags to train and predict Consumer Electronics named entities
in text.

src/sameword/
=============
A simple tool to detect word equivalences using Wordnet. Reads a TSV file of word pairs and returns the original (LHS) word if the words don't have the same meaning. Useful (at least in my case) for checking the results of a set of regular expressions to convert words from British to American spellings and for converting Greek/Latin plurals to their singular form (both based on patterns).

src/genetagger
==============
A Named Entity Recognizer for Genes - uses NLTK's HMM package to build an HMM tagger to recognize Gene names from within English text.

