# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 22:12:12 2019

@author: Air Junda
"""
import data_process as dp
import md_tuning as mt


import numpy as np
import sklearn
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.dummy import DummyClassifier


def docFilter(docs):    
    stoplist = set(nltk.corpus.stopwords.words())
    filtered_docs = []
    for doc in docs:
        tokens = nltk.word_tokenize(doc.lower())
        filtered_tokens = []
        for word in tokens:
            if word not in stoplist:
                filtered_tokens.append(word)
        filtered_docs.append(filtered_tokens)
    return filtered_docs

def convert_tag(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, convert_tag(w)) for w in tokens]

# transform the tokens into sklearn vectors
def dummy(doc):
    """This is the dummy tokenizer for CountVectorizer,
    because we already have the tokens for the reviews"""
    return doc

def get_vector(train, test, cv):
    """This method returns the vector for the train set and test set"""
    docs_train_vec = cv.fit_transform(train)
    docs_test_vec = cv.transform(test)
    return docs_train_vec, docs_test_vec

def doc2Vector(docs,cv):
    """This method returns the vector forms of the input documents """
    docvec = cv.fit_transform(docs)
    return docvec

def baseline(x_data, y_data, stra = "uniform"):
    """ baseline prediction using dummyClassifier """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    dummy = sklearn.dummy.DummyClassifier(strategy= stra)
    dummy.fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    accu = metrics.accuracy_score(y_test, y_pred)
    return accu


if __name__ == '__main__':
    """Run this file will print out the accuracy of two baseline methods
    """
    
    
    
    df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
    df = df.drop(df.columns[[2, 3, 4]], axis = 1) 
    targets = list(df['v1'])
    docs = list(df['v2'])
    filtered_docs  = docFilter(docs)
    lemmatized = [lemmatize(tokens) for tokens in filtered_docs]
    
    cv = CountVectorizer(min_df=2, tokenizer=dummy, preprocessor=dummy, max_features=5000)
    lemaed_docs  = doc2Vector(lemmatized,cv)  # lemmatized vectors
    cv_transformer = TfidfTransformer()
    
    docs_tfidf =  cv_transformer.fit_transform(lemaed_docs)  
    docs_tfidf = docs_tfidf.toarray()
    
    bl_unif = baseline(docs_tfidf, targets, stra = "uniform")
    bl_mostf = baseline(docs_tfidf, targets, stra = "most_frequent")
    
    print("Uniform baseline give accuracy of " + str(round(bl_unif,4)))
    print("Most frequent baseline give accuracy of " + str(round(bl_mostf,4)))

    

