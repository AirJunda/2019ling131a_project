import re
import pandas
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math

"""
This module implements the data reading and processing method such as tokenizer and remove stop words.
And also self-implemented bigram and positive mutual information bigram method for more choices of feature extraction.
"""


def read_data(path):
    df = pandas.read_csv(path, encoding="ISO-8859-1")
    targets = list(df['v1'])
    docs = list(df['v2'])
    return targets, docs


def my_tokenizer(raw):
    """Implement a raw text tokenizer using the regular expression. Note that the order of the patterns matters here."""
    return re.findall(
        r"""(
            \w+         # word characters
            |\.\.\.     # elipsis
            |\n{2,}     # white lines
            |-+         # dashes
            |[(){}[\]]  # brackets
            |[!?.,:;]   # punctuation
            |['"`]      # quotes
            |\S+        # fall through pattern
            )""",
        raw, re.VERBOSE)


def remove_stop_words(docs):
    """This method converts the text into lowercase and remove the stop words from the text"""
    stoplist = set(nltk.corpus.stopwords.words())
    filtered_docs = []
    for doc in docs:
        tokens = my_tokenizer(doc.lower())
        filtered_tokens = []
        for word in tokens:
            if word not in stoplist:
                filtered_tokens.append(word)
        filtered_docs.append(filtered_tokens)
    return filtered_docs


def lemmatize(tokens):
    """Wordnet is one of the earliest and most commonly used lemmatizers. We can pass in the Wordnet tag as an
    argument to tell the lemmatizer which context this word belongs. Pos_tag function in nltk assigns each word with
    POS tags, so we need to write a function to convert the pos_tag to wordnet tags. """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, convert_tag(w)) for w in tokens]


def convert_tag(word):
    """This method convert the tag into wordnet tags"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Optimazation via feature eng
def data_filter(docs):
    """This method filter the text by removing the stop words and lemmatizes it, and generate a unigram tokens set"""
    filter_stop_words = remove_stop_words(docs)
    lemmatized = [lemmatize(tokens) for tokens in filter_stop_words]
    return lemmatized


def data_filter_bigram(docs):
    """This method filter the text by removing the stop words and lemmatizes it, and generate a bigram tokens set"""
    lemmatized = data_filter(docs)
    bigrams = [get_bigram(doc) for doc in lemmatized]
    return bigrams


def get_bigram(doc):
    """This helper method generate bigrams for a unigram token set"""
    bigram_tuple = [b for b in zip(doc[:-1], doc[1:])]
    bigrams = []
    for t in bigram_tuple:
        bigrams.append(t[0] + ' ' + t[1])
    return bigrams


def frequency(docs):
    """Build a frequency map for the input text tokens"""
    fdist = defaultdict(int)
    for doc in docs:
        for token in doc:
            fdist[token] += 1
    return fdist


def pmi(bigram, fre_uni, fre_big):
    """calculate the mutual information for bigrams"""
    lst = bigram.split(" ")
    word1 = lst[0]
    word2 = lst[1]
    p1 = fre_uni[word1] / float(sum(fre_uni.values()))
    p2 = fre_uni[word2] / float(sum(fre_uni.values()))
    p12 = fre_big[bigram] / float(sum(fre_big.values()))
    return math.log(p12/float(p1*p2),2)


def filter_pmi(doc, fre_uni, fre_big):
    """filter all the bigrams with negative mutual information"""
    res = []
    for bigram in doc:
        if pmi(bigram, fre_uni, fre_big) > 0:
            res.append(bigram)
    return res