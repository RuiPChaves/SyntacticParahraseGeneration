# SyntacticParahraseGeneration

Syntactic Paraphrase generator engine from the paper

Chaves, Rui P. and Stephanie Richter 2021 "Look at that! BERT can be easily distracted from paying attention to morphosyntax", in 4th Annual Meeting of the Society for Computation in Linguistics (SCiL).
[https://www.acsu.buffalo.edu/~rchaves/AttentionBERT.pdf]

The Python3 code at sentence_generator.py and associated libraries was written by Stephanie Richter, to randomly generate clusters of sentences with various syntactic transformations, spread across two files (train and test, which do not share nouns and verbs). R code at format_items.r was written by Rui Chaves to randomly select pairs of sentences from clusters to create labeled paraphrase and non-paraphrase pairs ready for BERT fine-tuning and testing.

To run, first execute python code and then run R script on the output (train/test) csv files.

Example train/test outputs are in train.csv and test.csv. Sentences with the same "dyad" number are variants of each other. 
The train_r/test_r files are random pairings of paraphrase/non-paraphrase pairings from the above datasets.
