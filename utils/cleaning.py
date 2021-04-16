import pandas as pd
from string import punctuation
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
import spacy
from spacy_iwnlp import spaCyIWNLP
nlp=spacy.load('de')
iwnlp=spaCyIWNLP(lemmatizer_path='data/IWNLP.Lemmatizer_20181001.json')
nlp.add_pipe(iwnlp)
stemmer = SnowballStemmer('german')


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


def stem_germ(x):
    '''stemms german texts
    Arguments: x - a string containing german text
    Return : stemmed - the stemmed german text'''
    tok = nltk.word_tokenize(normalize(x))
    stemmed = " ".join([stemmer.stem(word) for word in tok])
    
    return stemmed

def lem_germ(x):
    '''lemmatizes german texts
    Arguments: x - a string containing german text
    Return : lemmed - the lemmatized german text'''
    tok = nlp(normalize(x))
    lemmed=""
    for word in tok:
        try:
            lemmed+=" " + word._.iwnlp_lemmas[0]
        except:
            lemmed+=" " + str(word)
    return lemmed

def lem_stem(series, lem_stem):
    '''stemms or lemmatizes (or both) a series of german texts.
    Arguments: series - a pandas series containing german texts.
                lem_stem - option wether to stem (="stem") to lemmatize (="lem") or to lemmatize then stem (="lem_stem")
    Return: new series - a stemmed or lemmatized (or both) series'''
    
    if lem_stem == 'stem':
        new_series = series.apply(stem_germ)
    elif lem_stem == 'lem':
        new_series = series.apply(lem_germ)
    elif lem_stem == 'lem_stem':
        new_series = series.apply(lem_germ).apply(stem_germ)
    return new_series

def normalize(txt, url_emoji_dummy=False):
    txt = txt.lower()

    url_dummy = ' '
    emoji_dummy = ' '
    if url_emoji_dummy:
        url_dummy = 'URL'
        emoji_dummy = 'EMOJI'
    # replace URLs
    # URLs starting with http(s) or ftp(s)
    url_re1 = re.compile(r'(?:ftp|http)s?://[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re1.sub(url_dummy, txt)
    # URLs starting with www.example.com
    url_re2 = re.compile(r'\bwww\.[a-zA-Z0-9-]{2,63}\.[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re2.sub(url_dummy, txt)
    # URLs short version example.com 
    url_re3 = re.compile(r'\b[a-zA-Z0-9.]+\.(?:com|org|net|io)')
    txt = url_re3.sub(url_dummy, txt)

    # replace emoticons
    # "Western" emoticons such as =-D and (^:
    s = r"(?:^|(?<=[\s:]))"      # beginning or whitespace required before
    s += r"(?:"                  # begin emoticon
    s += r"(?:"                  # begin "forward" emoticons like :-)
    s += r"[<>]?"                # optinal hat/brow
    s += r"[:;=8xX]"             # eyes
    s += r"[o*'^-]?"             # optional nose
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r")"                    # end "forward" emoticons
    s += r"|"                    # or
    s += r"(?:"                  # begin "backward" emoticons like (-:
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r"[o*'^-]?"             # optional nose
    s += r"[:;=8xX]"             # eyes
    s += r"[<>]?"                # optinal hat/brow
    s += r")"                    # end "backward" emoticons
    # "Eastern" emoticons like ^^ and o_O
    s += r"|"                    # or
    s += r"(?:\^\^)|(?:o_O)"     # only two eastern emoticons for now
    s += r")"                    # end emoticon
    s += r"(?=\s|$)"             # white space or end required after
    emoticon_re = re.compile(s)
    txt = emoticon_re.sub(emoji_dummy, txt)  #replace with 'EMOTICON but keep preceeding and trailing space/linefeed

    # replace punctuation by space
    txt = txt.translate({ord(c): " " for c in punctuation})

    # remove leading, trailing and repeated whitespace
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)

    return txt