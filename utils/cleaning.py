import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
stopwords=stopwords.words('german')

def strip_punct(series):
    '''Strip puncation of series of strings
    Arguments: series - a series containing strings
    Return: new-series - a series of strings without punctuation'''
    new_series = series.str.replace(r'[^\w\s]+', '', regex=True)
    return new_series


def strip_stopwords(series, stopwords=stopwords):
    '''Strip stopwords of series of strings
    Arguments: series - a series containing strings, stopwords - a list of stopwords (default: german)
    Return: new-series - a series of strings without stopwords'''
    series=series.copy()
    new_series = series.apply(lambda x: " ".join([word.lower() for word in x.split() if word.lower() not in (stopwords)]) if x is not None else x)
    return new_series

