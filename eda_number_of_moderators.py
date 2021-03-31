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

# %%
df = loading.load_extended_posts()

# %%
df.query('is_staff == 1').id_user.nunique()

# %%
df.query('is_staff == 1').shape[0]

# %%
df.query('is_staff == 1').id_user.value_counts().describe()

# %% [markdown]
# There are 110 moderators with 2179 posts. The moderators have written between 1 and 266 posts with a median of 6 and a mean of 19.8.
