# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from utils import loading
import pandas as pd
from nltk.corpus import stopwords
stopwords=stopwords.words('german')
from string import punctuation
from collections import Counter
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

# %%
df = loading.load_extended_posts()


# %% [markdown]
# Defining functions for data cleaning and plotting

# %%
# strip punctuation
def strip_punct(series):
    new_series = series.str.replace(r'[^\w\s]+', '', regex=True)
    return new_series


# %%
def strip_stopwords(series, stopwords=stopwords):
    series=series.copy()
    new_series = series.apply(lambda x: " ".join([word.lower() for word in x.split() if word.lower() not in (stopwords)]) if x is not None else x)
    return new_series


# %%
def get_top_words(series, relative=False):
    topwords=pd.Series(' '.join(series[(series==series)]).lower().split()).value_counts()
    if relative:
        return topwords/len(series)
    else:
        return topwords


# %%
def plot_wordcloud_series(series, colormap='BuGn'):
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate_from_frequencies(series) 
    wordcloud.recolor(colormap=colormap)    
    # plot the WordCloud image                        
    #plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    #plt.show()



# %%
def top_words_label(df, label, text, stop=False, stopwords=None, plot=True, return_list=True):
    
    df_clean=df.dropna(subset=[label])
    df_clean.loc[:,text]=strip_punct(df_clean[text])
    if stop:
        df_clean.loc[:,text]=strip_stopwords(df_clean[text], stopwords=stopwords)
    df_pos = df_clean[df_clean[label]==1]
    df_neg = df_clean[df_clean[label]==0]
    topwords_pos = get_top_words(df_pos[text], relative=True)
    topwords_neg = get_top_words(df_neg[text], relative=True)
    topwords_pos_rel = topwords_pos.subtract(topwords_neg, fill_value=0).sort_values(ascending=False)
    topwords_neg_rel = (-topwords_pos_rel).sort_values(ascending=False)
    if plot:
        print(f'Order of plots:\nTop left: {label} = positive\nTop right: {label} = negative\nBottom left: {label} = positive, specific\nBottom right: {label} = negative, specific')
        plt.figure(figsize = (12, 12))
        plt.subplot(2, 2, 1)
        plot_wordcloud_series(topwords_pos, colormap='BuGn')
        plt.subplot(2, 2, 2)
        plot_wordcloud_series(topwords_neg, colormap='RdPu')
        plt.subplot(2, 2, 3)
        plot_wordcloud_series(topwords_pos_rel,colormap='YlGn')
        plt.subplot(2, 2, 4)
        plot_wordcloud_series(topwords_neg_rel, colormap='OrRd')
        plt.show()
    if return_list:
        return topwords_pos, topwords_neg, topwords_pos_rel, topwords_neg_rel


# %% [markdown]
# # Getting the top words in comments for every label

# %% [markdown]
# ## Arguments used

# %%
arg_pos, arg_neg, arg_pos_rel, arg_neg_rel = top_words_label(df, 'label_argumentsused', 'body', True, stopwords)

# %% tags=[]
print(f'top words for argument used positive:\n{arg_pos[:10]}')
print(f'top words for argument used negative:\n{arg_neg[:10]}')
print(f'top words for argument used positive specific:\n{arg_pos_rel[:10]}')
print(f'top words for argument used negative specific:\n{arg_neg_rel[:10]}')

# %% [markdown]
# ## Discriminating

# %%
dis_pos, dis_neg, dis_pos_rel, dis_neg_rel = top_words_label(df, 'label_discriminating', 'body', True, stopwords)

# %%
print(f'top words for discriminating positive:\n{dis_pos[:10]}')
print(f'top words for discriminating negative:\n{dis_neg[:10]}')
print(f'top words for discriminating positive specific:\n{dis_pos_rel[:10]}')
print(f'top words for discriminating negative specific:\n{dis_neg_rel[:10]}')

# %% [markdown]
# ## Inappropriate

# %%
ina_pos, ina_neg, ina_pos_rel, ina_neg_rel = top_words_label(df, 'label_inappropriate', 'body', True, stopwords)

# %%
print(f'top words for innapropriate positive:\n{ina_pos[:10]}')
print(f'top words for innapropriate negative:\n{ina_neg[:10]}')
print(f'top words for innapropriate positive specific:\n{ina_pos_rel[:10]}')
print(f'top words for innapropriate negative specific:\n{ina_neg_rel[:10]}')

# %% [markdown]
# ## Off-Topic

# %%
ot_pos, ot_neg, ot_pos_rel, ot_neg_rel = top_words_label(df, 'label_offtopic', 'body', True, stopwords)

# %%
print(f'top words for Off-Topic positive:\n{ot_pos[:10]}')
print(f'top words for Off-Topic negative:\n{ot_neg[:10]}')
print(f'top words for Off-Topic positive specific:\n{ot_pos_rel[:10]}')
print(f'top words for Off-Topic negative specific:\n{ot_neg_rel[:10]}')

# %% [markdown]
# ## Personal stories

# %%
ps_pos, ps_neg, ps_pos_rel, ps_neg_rel = top_words_label(df, 'label_personalstories', 'body', True, stopwords)

# %%
print(f'top words for Personal Stories positive:\n{ps_pos[:10]}')
print(f'top words for Personal Stories negative:\n{ps_neg[:10]}')
print(f'top words for Personal Stories positive specific:\n{ps_pos_rel[:10]}')
print(f'top words for Personal Stories negative specific:\n{ps_neg_rel[:10]}')

# %% [markdown]
# ## Possibly Feedback

# %%
fb_pos, fb_neg, fb_pos_rel, fb_neg_rel = top_words_label(df, 'label_possiblyfeedback', 'body', True, stopwords)

# %%
print(f'top words for Possibly Feedback positive:\n{fb_pos[:10]}')
print(f'top words for Possibly Feedback negative:\n{fb_neg[:10]}')
print(f'top words for Possibly Feedback positive specific:\n{fb_pos_rel[:10]}')
print(f'top words for Possibly Feedback negative specific:\n{fb_neg_rel[:10]}')

# %% [markdown]
# ## Sentiment
# ### Negative

# %%
sng_pos, sng_neg, sng_pos_rel, sng_neg_rel = top_words_label(df, 'label_sentimentnegative', 'body', True, stopwords)

# %%
print(f'top words for Sentiment Negative positive:\n{sng_pos[:10]}')
print(f'top words for Sentiment Negative negative:\n{sng_neg[:10]}')
print(f'top words for Sentiment Negative positive specific:\n{sng_pos_rel[:10]}')
print(f'top words for Sentiment Negative negative specific:\n{sng_neg_rel[:10]}')

# %% [markdown]
# ### Neutral

# %%
snt_pos, snt_neg, snt_pos_rel, snt_neg_rel = top_words_label(df, 'label_sentimentneutral', 'body', True, stopwords)

# %%
print(f'top words for Sentiment Neutral positive:\n{snt_pos[:10]}')
print(f'top words for Sentiment Neutral negative:\n{snt_neg[:10]}')
print(f'top words for Sentiment Neutral positive specific:\n{snt_pos_rel[:10]}')
print(f'top words for Sentiment Neutral negative specific:\n{snt_neg_rel[:10]}')

# %% [markdown]
# ### Positive

# %%
spo_pos, spo_neg, spo_pos_rel, spo_neg_rel = top_words_label(df, 'label_sentimentpositive', 'body', True, stopwords)

# %%
print(f'top words for Sentiment Positive positive:\n{spo_pos[:10]}')
print(f'top words for Sentiment Positive negative:\n{spo_neg[:10]}')
print(f'top words for Sentiment Positive positive specific:\n{spo_pos_rel[:10]}')
print(f'top words for Sentiment Positive negative specific:\n{spo_neg_rel[:10]}')

# %% [markdown]
# # Getting the top words in headline for every label

# %% [markdown]
# ## Arguments Used

# %%
arg_pos, arg_neg, arg_pos_rel, arg_neg_rel = top_words_label(df, 'label_argumentsused', 'headline', True, stopwords)

# %% tags=[]
print(f'top words for argument used positive:\n{arg_pos[:10]}')
print(f'top words for argument used negative:\n{arg_neg[:10]}')
print(f'top words for argument used positive specific:\n{arg_pos_rel[:10]}')
print(f'top words for argument used negative specific:\n{arg_neg_rel[:10]}')

# %% [markdown]
# ## Discriminating

# %%
dis_pos, dis_neg, dis_pos_rel, dis_neg_rel = top_words_label(df, 'label_discriminating', 'headline', True, stopwords)

# %%
print(f'top words for discriminating positive:\n{dis_pos[:10]}')
print(f'top words for discriminating negative:\n{dis_neg[:10]}')
print(f'top words for discriminating positive specific:\n{dis_pos_rel[:10]}')
print(f'top words for discriminating negative specific:\n{dis_neg_rel[:10]}')

# %% [markdown]
# ## Inappropriate

# %%
ina_pos, ina_neg, ina_pos_rel, ina_neg_rel = top_words_label(df, 'label_inappropriate', 'headline', True, stopwords)

# %%
print(f'top words for innapropriate positive:\n{ina_pos[:10]}')
print(f'top words for innapropriate negative:\n{ina_neg[:10]}')
print(f'top words for innapropriate positive specific:\n{ina_pos_rel[:10]}')
print(f'top words for innapropriate negative specific:\n{ina_neg_rel[:10]}')

# %% [markdown]
# ## Off-Topic

# %%
ot_pos, ot_neg, ot_pos_rel, ot_neg_rel = top_words_label(df, 'label_offtopic', 'headline', True, stopwords)

# %%
print(f'top words for Off-Topic positive:\n{ot_pos[:10]}')
print(f'top words for Off-Topic negative:\n{ot_neg[:10]}')
print(f'top words for Off-Topic positive specific:\n{ot_pos_rel[:10]}')
print(f'top words for Off-Topic negative specific:\n{ot_neg_rel[:10]}')

# %% [markdown]
# ## Personal stories

# %%
ps_pos, ps_neg, ps_pos_rel, ps_neg_rel = top_words_label(df, 'label_personalstories', 'headline', True, stopwords)

# %%
print(f'top words for Personal Stories positive:\n{ps_pos[:10]}')
print(f'top words for Personal Stories negative:\n{ps_neg[:10]}')
print(f'top words for Personal Stories positive specific:\n{ps_pos_rel[:10]}')
print(f'top words for Personal Stories negative specific:\n{ps_neg_rel[:10]}')

# %% [markdown]
# ## Possibly Feedback

# %%
fb_pos, fb_neg, fb_pos_rel, fb_neg_rel = top_words_label(df, 'label_possiblyfeedback', 'headline', True, stopwords)

# %%
print(f'top words for Possibly Feedback positive:\n{fb_pos[:10]}')
print(f'top words for Possibly Feedback negative:\n{fb_neg[:10]}')
print(f'top words for Possibly Feedback positive specific:\n{fb_pos_rel[:10]}')
print(f'top words for Possibly Feedback negative specific:\n{fb_neg_rel[:10]}')

# %% [markdown]
# ## Sentiment
# ### Negative

# %%
sng_pos, sng_neg, sng_pos_rel, sng_neg_rel = top_words_label(df, 'label_sentimentnegative', 'headline', True, stopwords)

# %%
print(f'top words for Sentiment Negative positive:\n{sng_pos[:10]}')
print(f'top words for Sentiment Negative negative:\n{sng_neg[:10]}')
print(f'top words for Sentiment Negative positive specific:\n{sng_pos_rel[:10]}')
print(f'top words for Sentiment Negative negative specific:\n{sng_neg_rel[:10]}')

# %% [markdown]
# ### Neutral

# %%
snt_pos, snt_neg, snt_pos_rel, snt_neg_rel = top_words_label(df, 'label_sentimentneutral', 'headline', True, stopwords)

# %%
print(f'top words for Sentiment Neutral positive:\n{snt_pos[:10]}')
print(f'top words for Sentiment Neutral negative:\n{snt_neg[:10]}')
print(f'top words for Sentiment Neutral positive specific:\n{snt_pos_rel[:10]}')
print(f'top words for Sentiment Neutral negative specific:\n{snt_neg_rel[:10]}')

# %% [markdown]
# ### Positive

# %%
spo_pos, spo_neg, spo_pos_rel, spo_neg_rel = top_words_label(df, 'label_sentimentpositive', 'headline', True, stopwords)

# %%
print(f'top words for Sentiment Positive positive:\n{spo_pos[:10]}')
print(f'top words for Sentiment Positive negative:\n{spo_neg[:10]}')
print(f'top words for Sentiment Positive positive specific:\n{spo_pos_rel[:10]}')
print(f'top words for Sentiment Positive negative specific:\n{spo_neg_rel[:10]}')

