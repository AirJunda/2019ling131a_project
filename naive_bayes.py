import data_process
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

"""This module uses methods in data_process to process the dataset and trains the Naive Bayes classifier:
1. Tf-idf unigram model: using filtered unigram data
2. Tf-idf bigram model: using filtered bigram data
3. Tf-idf bigram pmi model: based on 2, only keep the bigrams with positive mutual information as feature set
"""
# split the data into training and test sets
targets, raw_docs = data_process.read_data("data/spam.csv")
docs = data_process.data_filter(raw_docs)
docs_train, docs_test, cate_train, cate_test = train_test_split(docs, targets, test_size=0.20,
                                                                random_state=12)
# training and test sets for bigram model
docs_bigram_train = [data_process.get_bigram(doc) for doc in docs_train]
docs_bigram_test = [data_process.get_bigram(doc) for doc in docs_test]
# using bigram with positive mutual information as training set
fre_uni = data_process.frequency(docs_train)
fre_big = data_process.frequency(docs_bigram_train)
docs_bigram_train_pmi = [data_process.filter_pmi(doc, fre_uni, fre_big) for doc in docs_bigram_train]


def dummy(doc):
    """This is the dummy tokenizer for CountVectorizer,
    because we already have the tokens for the reviews"""
    return doc


def get_vector(train, test, cv):
    """This method returns the vector for the train set and test set"""
    docs_train_vec = cv.fit_transform(train)
    docs_test_vec = cv.transform(test)
    return docs_train_vec, docs_test_vec


def classifier_training(train, test, min_count):
    """This method use the provided tokens to train the classifier"""
    # initialize CountVectorizer
    cv = CountVectorizer(min_df=min_count, tokenizer=dummy, preprocessor=dummy, max_features=5000)
    # fit and transform the train set and test set
    docs_train_vec, docs_test_vec = get_vector(train, test, cv)
    # Convert raw frequency counts into TF-IDF values
    cv_transformer = TfidfTransformer()
    docs_train_tfidf = cv_transformer.fit_transform(docs_train_vec)
    docs_test_tfidf = cv_transformer.fit_transform(docs_test_vec)
    # use multinominal Naive Bayes as our model
    classifier = MultinomialNB()
    # training the Naive Bayes classifier
    classifier.fit(docs_train_tfidf, cate_train)
    return round(classifier.score(docs_test_tfidf, cate_test), 3)


if __name__ == '__main__':
    print("Accuracy using tf-idf with filtered words: ")
    print(classifier_training(docs_train, docs_test, 2))
    print("Accuracy using tf-idf with filtered words bigram: ")
    print(classifier_training(docs_bigram_train, docs_bigram_test, 5))
    print("Accuracy using tf-idf with filtered words pmi bigram: ")
    print(classifier_training(docs_bigram_train_pmi, docs_bigram_test, 5))
