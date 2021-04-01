import pandas as pd
def calculate_top_words(series, relative=False):
    '''calculate word frequencies from a series of strings
    Arguments: series - a series containing strings, relative - If true calculate proportion of word frequency by number of observation, if False calculate absolute word frequency
    Return: topwords - a series containing word frequencies'''
    topwords=pd.Series(' '.join(series[(series==series)]).lower().split()).value_counts()
    if relative:
        return topwords/len(series)
    else:
        return topwords