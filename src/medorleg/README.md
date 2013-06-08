INTRODUCTION

The motivation for this project is to automatically determine if a sentence is from a medical or a legal genre. This is needed to support a medical litigation decision support system which allows a user to search for documents using concepts rather than terms[1]. Concepts are medical and legal named entities, such as diseases, drugs, organizations, jurisdictions, etc. These entities are extracted during document indexing by looking up words and phrases in each sentence against domain specific taxonomies for each genre. Documents are then annotated with the entity IDs, where they can be discovered during concept search. Unfortunately, this can often produce ambiguous concepts that can mean very different things in the two genres - for example, hearing, period and immunity. This creates a bad search experience.

To address this, we build a classifier that can classify an incoming sentence into one of the two genres. This classifier acts as a preprocessor that will route the sentence into one of two knowledge bases for entity recognition and annotation.

METHODS

We build two interpolated trigram language models[2], one for each genre. An interpolated trigram language model approximates the probability of a specific trigram as a linear combination of the frequency of the trigram and the associated bigram and unigram. The interpolated score for each trigram (w1,w2,w3) is given by:

    p(w1,w2,w3|c) = α * p(w1,w2,w3) + β * p(w2,w3) + γ * p(w3)    ...(1)
    where:
      p(w1,w2,w3|c) = C(w1,w2,w3) / Number of trigrams in corpus
      p(w1,w2,w3)   = C(w1,w2,w3) / C(w2,w3)
      p(w2,w3)      = C(w2,w3) / C(w3)
      p(w3)         = (C(w3) + 1) / (N + V)
      N             = number of terms in corpus
      V             = number of vocabulary terms
The unigram probability is smoothed using Laplace smoothing so unseen unigrams in the test set don't result in a zero score for the sentence. The parameters α, β and γ are learned from the training data using a simple linear regression algorithm.

The coefficients of the linear model should also satisfy this constraint:

    α + β + γ = 1                    ...(2)
Using the models for the two genres as described above, the probability for an unseen sentence is calculated for each genre, as the joint probability (product) of each of its trigrams. The sentence is classified into the genre which is more probable according to the model.

ANALYSIS

The model described above was generated and tested as follows:

Our training set consists of a medical corpus (about 10,000 files from the Gale Medical Encyclopedia[3]), and a legal corpus (about 4,000 files from the Australian Case Law Dataset[4]). The first step (preprocess.py[7]) is to parse the XML files in each corpus into a single flat file of sentences. Each sentence is tagged with a source (M or L). We end up with 950,887 medical sentences and 837,393 legal sentences.
We randomly choose 1,000 sentences from each category (testset_splitter.py[7]) to use as a test set for evaluating our classifier.
Use a MapReduce job (ngram_counting_job.py[7]) using mrjob[5] and NLTK to compute the counts for trigrams, bigrams and unigrams for each sentence. This gives us 3,740,646 medical and 5,092,913 legal count records.
Populate a SQLite3 database with the aggregated counts (db_loader.py[7]). We use SQLite3 as a persistent lookup table for doing probability calculations (regression_data.py[7]) for each trigram, ie finding the p(w) values in equation (1) above. The output of this step is a set of X (variable) and y (outcome) values for the trigrams in each genre.
Train a Linear Regression model for each genre (model_params.py[7]). The coefficients of the Linear model correspond to the unnormalized values of α, β, and γ in equation (2). We normalize both models by dividing the coefficients and the intercept by the sum of the coefficients.
Convert each test sentences into trigrams and compute the joint probability of the trigrams against each model (eval_model.py[7]), and report on overall accuracy of the model.
RESULTS

The overall accuracy for the classifier was 92.7%. The legal model performed better, correctly classifying 997 of 1,000 legal documents, as opposed to the medical model, which correctly classfied only 857 of 1,000 medical documents. The confusion matrix is shown below:

            M     L  <-- classified as
          857   143 |   M
            3   997 |   L

CONCLUSION

An accuracy of 92.7% is adequate for our purposes, so we can consider putting this model into production. However, its performance can probably be improved through the use of more sophisticated regression algorithms.

REFERENCES

[1] Concept Search [http://en.wikipedia.org/wiki/Concept_Search, downloaded May 30, 2013].
[2] Interpolated Language Models from Introduction to Information Retrieval Chapter 12: Language Models for Information Retrieval, by Manning, Schultz, et al [http://nlp.stanford.edu/IR-book/pdf/12lmodel.pdf, downloaded May 30, 2013].
[3] Gale Encyclopedia of Medicine, 4/Ed (EBook Edition) [http://www.gale.cengage.com/servlet/ItemDetailServlet?region=9&imprint=000&cf=e&titleCode=GEME&type=4&id=259611, downloaded May 31, 2013].
[4] UCI Machine Learning Repository Legal Case Reports Dataset [http://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports, downloaded May 25, 2013].
[5] MRJob Tutorial [http://www.brianweidenbaum.com/mapreduce-python-mrjob-tutorial/, downloaded May 31, 2013].
[6] R-Squared Coefficient of Determination [http://en.wikipedia.org/wiki/Coefficient_of_determination, downloaded June 3, 2013].
[7] Code for the analysis [https://github.com/sujitpal/nltk-examples/tree/master/src/medorleg, uploaded June 3, 2013].
[8] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. 
