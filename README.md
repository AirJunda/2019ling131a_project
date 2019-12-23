# Spam SMS Message Classification: A Naïve Bayes Approach

This is the repository for the final project of course LING131A for 2019Fall.

Group member: Junda Li && Tongkai Zhang(tongkaizhang@brandeis.edu)

This project is about applying Naive bayes method in classification of Spam SMS messages.

Dependencies:
1. pandas
2. nltk
3. sklearn
4. pickle

Instruction:

data_process.py: This module implements the data reading and processing method such as tokenizer and remove stop words.
And also self-implemented bigram and positive mutual information bigram method for more choices of feature extraction.

Baseline.py : This file generates baseline results of spam classification. The result will be print out in console.

naive_bayes.py: This module uses methods in data_process to process the dataset and trains the Naive Bayes classifier:
1. Tf-idf unigram model: using filtered unigram data
2. Tf-idf bigram model: using filtered bigram data
3. Tf-idf bigram pmi model: based on 2, only keep the bigrams with positive mutual information as feature set
How to run this: in the terminal 
  'python3 naive_bayes.py --train' : train these three models
  'python3 naive_bayes.py --run This is a sample message' : predict the result
Note that according to our experiment on the feature set optimization, the unigram model has highest accuracy, so here we use unigram classifier to predict
  

md_tuning.py : This file searches best alpha value for the naïve bayes method and reports the false positive rate of tuning and untuning Naive Bayes models. The whole file is divided into cells. Each cell start with #%%. To run this file, you can run the whole the file in one fire or run each cell one by one in Spyder IDE( Similar to run each code block in Jupyter notebook). The result will be printed out in console.

