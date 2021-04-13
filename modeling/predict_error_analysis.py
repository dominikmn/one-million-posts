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
# # Script for error analysis
#
# + Loads saved mlflow-model from directory `"<folder-name>/<algorithm>_label_<label-name>_<...>"`
# + Predicts on validation set
# + Stores results in "./output/"
#
# Call with `python -m modeling.predict_error_analysis <folder-name>/<algorithm>_label_<label-name>_<...>`

# %%
import sys
import pandas as pd
import mlflow
from mlflow.sklearn import load_model
from utils import loading, feature_engineering

# %%
# get path to model from system arguments and extract label
model_path = sys.argv[1]
label = "_".join(model_path.split("_")[1:3])

# load the model from disk
loaded_model = load_model(model_path)

# load validation set and prepare features and target
df_val = loading.load_extended_posts(split="val")
df_val = feature_engineering.add_column_text(df_val)
X_val = df_val.text
y_val_true = df_val[label]

# predict
y_val_pred = loaded_model.predict(X_val)

# Misclassification
label_pred = f"{label}_pred"
df_val[label_pred] = y_val_pred
df_val["misclassified"] = df_val[label] - df_val[label_pred] # -1: false positive, 1: false negative

id_post_fn = df_val.query('misclassified == 1').id_post
id_post_fp = df_val.query('misclassified == -1').id_post
print(f"Number of FN: {id_post_fn.shape[0]}")
print(f"Number of FP: {id_post_fp.shape[0]}")

# save results
model_path = model_path.split("/")[1]
df_val[["id_post", label_pred]].to_csv(f"./output/{model_path}_pred.csv")
id_post_fn.to_csv(f"./output/misclassification_{model_path}_fn.csv")
id_post_fp.to_csv(f"./output/misclassification_{model_path}_fp.csv")

# %%
