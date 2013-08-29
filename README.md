nltk-examples
=============

## src/book/

Worked examples from the NLTK Book

## src/cener/

A Consumer Electronics Named Entity Recognizer - uses an NLTK Maximum Entropy
Classifier and IOB tags to train and predict Consumer Electronics named entities
in text.

## src/sameword/

A simple tool to detect word equivalences using Wordnet. Reads a TSV file of word pairs and returns the original (LHS) word if the words don't have the same meaning. Useful (at least in my case) for checking the results of a set of regular expressions to convert words from British to American spellings and for converting Greek/Latin plurals to their singular form (both based on patterns).

## src/genetagger/

A Named Entity Recognizer for Genes - uses NLTK's HMM package to build an HMM tagger to recognize Gene names from within English text.

## src/langmodel/

A trigram backoff language model trained on medical XML documents, and used to estimate the normalized log probability of an unknown sentence.

## src/docsim/

A proof of concept for calculating inter-document similarities for a collection of text documents for a cheating detection system. Contains implementation of the SCAM (Standard Copy Analysis Mechanism) in order to possible near-duplicate documents.

## src/phrases/

A proof of concept to identify significant word collocations as phrases from about an hours worth of messages from the Twitter 1% feed, calculated as a log-likelihood ratio of the probability that they are dependent vs that they are independent. Based on the approach described in "Building Search Applications: Lucene, LingPipe and GATE" by Manu Konchady, but extended to handle any size N-gram.

## src/medorleg

A trigram interpolated model trained on medical and legal sentences, and used to classify a sentence as one of the two genres.

## src/medorleg2

Uses the same training data as medorleg, but uses Scikit-Learn's text API and LinearSVC implementations to build a classifier that predicts the genre of an unseen sentence.

