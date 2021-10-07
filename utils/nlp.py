import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
import spacy
from spacy_iwnlp import spaCyIWNLP
nlp=spacy.load('de') #You need to download the 'de'-module in advance. This will be done automatically if you run `make setup` via the Makefile found in the main folder of the repo.
from pathlib import Path
path_here = Path(__file__).resolve().parent
iwnlp=spaCyIWNLP(lemmatizer_path= path_here / "../data/IWNLP.Lemmatizer_20181001.json")
nlp.add_pipe(iwnlp)
stemmer = SnowballStemmer('german')

from utils import cleaning

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
    tok = nltk.word_tokenize(cleaning.normalize(x))
    stemmed = " ".join([stemmer.stem(word) for word in tok])
    
    return stemmed

def lem_germ(x):
    '''lemmatizes german texts
    Arguments: x - a string containing german text
    Return : lemmed - the lemmatized german text'''
    tok = nlp(cleaning.normalize(x))
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
