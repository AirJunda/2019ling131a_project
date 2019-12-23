# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:34:08 2019

@author: Air Junda
"""
import numpy as np
import sklearn
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

#%%  
""" Load the dataset and split it into labels and SMS documents """
df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
df = df.drop(df.columns[[2, 3, 4]], axis = 1) 
targets = list(df['v1'])
docs = list(df['v2'])

#%% 
""" Filtered out stopwords  """
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

filtered_docs  = docFilter(docs)

#%%  
""" apply lemmatizaton on the filtered documents  """
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

lemmatized = [lemmatize(tokens) for tokens in filtered_docs]

#%%
""" Transformed each doucment into tf-idf arrays"""
cv = CountVectorizer(min_df=2, tokenizer=dummy, preprocessor=dummy, max_features=5000)
cv_transformer = TfidfTransformer()

lemaed_docs  = doc2Vector(lemmatized,cv)  # lemmatized vectors
docs_tfidf =  cv_transformer.fit_transform(lemaed_docs)
docs_tfidf = docs_tfidf.toarray()  # tf-idf vectors of all sms documents


#%% 
""" Grid search for best alpha value  """
def crossVal(clf,x_data, y_data,n):
    """return mean accuracy of Cross Validation """
    result = []
    for _ in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        clf.fit(x_train, np.ravel(y_train))
        y_pred = clf.predict(x_test)
        accu = metrics.accuracy_score(y_test, y_pred)
        result.append(accu)
    meanaccu = np.mean(result)
    std = np.std(result)    
    return (meanaccu,std)


def search(x_data, y_data, n = 5):
    """Apply grid-search for best alpha value"""
    alpha = np.arange(0.01, 4, 0.01)
    param_grid = {'alpha' : alpha}                
    clf = MultinomialNB()  
    grid_search = GridSearchCV(clf, param_grid, cv=n)
    grid_search.fit(x_data, y_data)
    return grid_search.best_params_

bestparms = search(docs_tfidf,targets,n = 2)
clfbest = MultinomialNB(alpha = bestparms['alpha'])
bestreport = crossVal(clfbest,docs_tfidf, targets,10)

print("The best alpha value is " + str(round(bestparms['alpha'],4)))
print("The accuracy of the NB with best alpha value is " + str(round(bestreport[0],4)))


#%%
""" Obatin false positive(fp) rate """
def cfmatrix(clf,x_data, y_data):
    ''' Obtain confusion matrix from a run of classification'''
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    clf.fit(x_train, np.ravel(y_train))
    y_pred = clf.predict(x_test)
    return metrics.confusion_matrix(y_test, y_pred,labels=["ham", "spam"])

def fp_rate(cfmatrix):
    ''' Calculation of false positve rate from confusion matrix'''
    fp = cfmatrix[0,1]/( cfmatrix[0,0] + cfmatrix[0,1])
    return fp

# calcuation of false positive rate of best NB model
def avg_fp(alp = 1):
    ''' Calculation of average false positve rate from confusion matrix for 10 repeated experiments'''
    fps = []
    for i in range(0,10):
        clf = MultinomialNB(alpha = alp)
        cmatrix = cfmatrix(clf,docs_tfidf, targets)
        fp_normal = fp_rate(cmatrix)
        fps.append(fp_normal)
    avg_fp  = np.mean(fps)
    return avg_fp

avgFp_noTuning = avg_fp()
avgFp_Tuning = avg_fp(alp = bestparms['alpha'])

print("The False Positive rate of NB without tuning is " + str(round(avgFp_noTuning,3)))
print("The False Positive rate of NB after tuning is " + str(round(avgFp_Tuning,3)))
    









