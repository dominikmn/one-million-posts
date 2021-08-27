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

# %% [markdown]
# # Eda issue 39 distribution of lables #39
# Issue link: https://github.com/dominikmn/one-million-posts/issues/39

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
# prepare extended annotation dataframe with duplicates annotated with "all"
df_ann_ext = df_ann.copy()
df_ann_ext.ann_round = "all"
df_ann.ann_round = df_ann.ann_round.astype(str)
df_ann_ext = pd.concat([df_ann_ext, df_ann], axis=0)
df_ann_ext.shape

# %%
# annotation counts per value, annotation round, and category
df_ann_counts = df_ann_ext.groupby(["category", "ann_round"])["value"].value_counts().unstack(level=2).unstack(level=1)
df_ann_counts


# %%

# %% [markdown]
# ## Bar chart
#
# For each label there are three bars indicating the percentage of positive lables annotated in this round. All means round 2 and 3.
# This plot shows a "true" label distribution for randomly sampled posts in round 2. The posts from round 3 are sampled in a way to increase positive annotations for rare labels.

# %%
# kindly borrowed and adjusted from https://stackoverflow.com/questions/35692781/python-plotting-percentage-in-seaborn-bar-plot
def add_annotation_with_hue(ax, absolute_values):
    """Add annotations to barplot with hue

    Args:
        ax: The axes of the plot.
        absolute_values: DataFrame with values to annotate
            (x lables of plot as index, hue lables as columns; the order must be equivalent to the order in the plot!).
    """
    labels = absolute_values.index.to_list()
    hue_categories = absolute_values.columns.to_list()
    number_of_categories = len(labels) 
    number_hue_categories = len(hue_categories)
    patch = [p for p in ax.patches]
    for i, label in enumerate(labels):
        for j, hue_column in enumerate(hue_categories):
            annotation = f'{absolute_values.loc[label, hue_column]:.0f}'
            x = patch[(j*number_of_categories + i)].get_x() + patch[(j*number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j*number_of_categories + i)].get_y() + patch[(j*number_of_categories + i)].get_height()  + 0.01
            ax.annotate(annotation, (x, y), size = 14)


# %%
font = {'size'   : 16}
matplotlib.rc('font', **font)

plt.figure(figsize=(12,6))
COLOR_STANDARD = ["#EC008E", "#bdbdbd", "#636363"]
g = sns.barplot(data=df_ann_ext, x="category", y="value", hue="ann_round", ci=None, hue_order=["all", "2", "3"], palette=sns.color_palette(COLOR_STANDARD))
sns.despine(left = True, bottom = True)
_ = g.set_title("Positive annotations per label and annotation round")
_ = g.set(ylabel="Proportion of positive annotations", xlabel="Label")
_ = g.legend(title="Annotation round")
_ = g.set_xticklabels(
    g.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right'
)
g.set_ylim(bottom=0, top=1);
add_annotation_with_hue(g, df_ann_counts.reindex(columns=["all", "2", "3"], level="ann_round").loc[:, 1])
plt.savefig("./pictures/positive_annotations_per_label_with_count.png", bbox_inches="tight")

# %% [markdown]
# ## Stacked Bar Chart
#
# Create one bar per label, showing the percentage of the postitive annotations of the label. The hue indicates the annotation round.
#
# This plot stresses how much of our data is "useless" due to the sampling strategy in annotation round 3.

# %%
# Seaborn does not support stacked bar charts. Therefore, I need to plot once the cumulative bar and the smaller one on top.
# Now, calculate the percentages of positive annotations from round 2 and all annotations with df_ann_counts
total = df_ann_counts.loc[:, (1, "all")] + df_ann_counts.loc[:, (0, "all")]
df_ann_perc_2 = df_ann_counts.loc[:, (1, "2")] / total
df_ann_perc_all = df_ann_counts.loc[:, (1, "all")] / total
df_ann_per = pd.concat({"2": df_ann_perc_2, "total": df_ann_perc_all}, axis=1).reset_index()
df_ann_per

# %%
f, ax = plt.subplots(figsize = (12,6))
g = sns.barplot(x = 'category', y = 'total', data = df_ann_per,
            label = '3', color = COLOR_STANDARD[2], edgecolor = 'w')
sns.set_color_codes('muted')
sns.barplot(x = 'category', y = '2', data = df_ann_per,
            label = '2', color = COLOR_STANDARD[0], edgecolor = 'w')
ax.legend(ncol = 2, loc = 'upper left')
sns.despine(left = True, bottom = True)
#_ = g.set_title("Positive annotations per label")
_ = g.set(ylabel="Proportion of positive annotations", xlabel="Label")
_ = g.legend(title="Annotation round")
_ = g.set_xticklabels(
    g.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right'
)
#g.set_ylim(bottom=0, top=1);
plt.savefig("./pictures/positive_annotations_per_label_stacked.png", bbox_inches="tight");

# %%
