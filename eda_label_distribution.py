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
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from utils import loading, feature_engineering

# %%
df_ann = loading.load_annotations()
df_ann = feature_engineering.add_column_ann_round(df_ann)
df_ann.head()

# %%
df_ann.ann_round.unique()

# %%
df_ann_ext = df_ann.copy()
df_ann_ext.ann_round = "all"
df_ann.ann_round = df_ann.ann_round.astype(str)
df_ann_ext = pd.concat([df_ann_ext, df_ann], axis=0)
df_ann_ext.shape


# %%
# todo: total and a need to be fixed!
# kindly borrowed from https://stackoverflow.com/questions/35692781/python-plotting-percentage-in-seaborn-bar-plot
def add_counts(ax, feature, number_of_categories, hue_categories):
    a = [p.get_height() for p in ax.patches]
    patch = [p for p in ax.patches]
    for i in range(number_of_categories):
        for j in range(hue_categories):
            total = feature.value_counts().values[j]
            percentage = '{:.1f}'.format(a[(j*number_of_categories + i)]*total)
            x = patch[(j*number_of_categories + i)].get_x() + patch[(j*number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j*number_of_categories + i)].get_y() + patch[(j*number_of_categories + i)].get_height() 
            ax.annotate(percentage, (x, y), size = 12)
    plt.show()


# %%
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

plt.figure(figsize=(12,6))
COLOR_STANDARD = ["#EC008E", "#bdbdbd", "#636363"]
g = sns.barplot(data=df_ann_ext, x="category", y="value", hue="ann_round", ci=None, hue_order=["all", "2", "3"], palette=sns.color_palette(COLOR_STANDARD))
_ = g.set_title("Positive annotations per label and annotation round")
_ = g.set(ylabel="Positive annotations [%]", xlabel="Label")
_ = g.legend(title="Annotation round")
_ = g.set_xticklabels(
    g.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right'
)
g.set_ylim(bottom=0, top=1);
plt.savefig("./pictures/positive_annotations_per_label.png", bbox_inches="tight")

# %%
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

plt.figure(figsize=(16,6))
COLOR_STANDARD = ["#bdbdbd", "#636363"]
g = sns.countplot(data=df_ann, x="category", hue="ann_round", hue_order=["2", "3"], palette=sns.color_palette(COLOR_STANDARD))
_ = g.set_title("Positive annotations per label and annotation round")
_ = g.set(ylabel="Positive annotations [%]", xlabel="Label")
_ = g.legend(title="Annotation round)")
_ = g.set_xticklabels(
    g.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right'
)
g.set_ylim(bottom=0, top=1);

# %%
import seaborn as sns
sns.set()
df_ann.groupby(["category", "ann_round"])["value"].value_counts()#set_index('ann_round').value_counts()#.plot(kind='bar', stacked=True)

# %%
