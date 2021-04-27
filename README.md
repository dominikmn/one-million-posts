# One Million Posts

Natural language processing project based on the [one-million-posts dataset](https://ofai.github.io/million-post-corpus/).

More than 3.000 user comments are written each day on www.derstandard.at. Moderators need to review these comments regarding several aspects like inappropriate language, discriminating content, off-topic comments, questions that need to be answered, and more.	

We provide machine learning models that detect potentially problematic comments to ease the moderators' daily work.

## Setup
1. Install [pyenv](https://github.com/pyenv/pyenv).
2. Install python 3.8.5 via `pyenv install 3.8.5`
3. Run `make setup`. 

### Setup - Modeling
See [SETUP.md](SETUP.md).

### Setup - Notebooks
The notebooks are pushed as `.py` files in the _python percentage script_ format (we like meaningful diffs).  
To get the actual notebook experience open them via jupyter with the [jupytext](https://github.com/mwouts/jupytext) plugin (gets installed as part of `make setup`).

### Setup - Dashboard
Starting the backend requires that a local model file is available.

Initial setup
1. Run `make dashboard`
2. Run `make backend`

Start backend
1. Open a dedicated terminal session.
2. Load the environment `source .venv_backend/bin/activate`
3. Start the backend via `uvicorn prediction_server:app --reload`

Start dashboard
1. Open a dedicated terminal session
2. Load the environment `source .venv_frontend/bin/activate` 
3. Start the dashboard via `python app.py` and copy the address+port that is displayed.
4. Open it in your browser like `http://127.0.0.1:8050/` (replace 8050 with the actual port displayed in step 3. above).

## Presentations
The presentations are found in `./presentations/`
| Presentation file | Description |
|-|-|
| [One Million Posts - Annotation composition.pdf](https://github.com/dominikmn/one-million-posts/blob/general-readme-update-midterm/presentations/One%20Million%20Posts%20-%20Annotation%20composition.pdf) | EDA concerning ticket [#24][i24], [#25][i25] |

[i24]: https://github.com/dominikmn/one-million-posts/issues/24
[i25]: https://github.com/dominikmn/one-million-posts/issues/25

## Modeling
The models' code is found in  `./modeling/` in this repo.
They are pushed as `.py` files. See [Setup - Modeling](#setup---modeling)
| Model | Description |
|-|-|
| [gbert Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/gbert_classifier.py) | [German BERT base](https://huggingface.co/deepset/gbert-base) | 
| [Zero Shot Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling_zero_shot.py) | [xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) |
| [XGBoost](https://github.com/dominikmn/one-million-posts/blob/main/modeling/xg_boost.py) | XGBoost |
| [Logistic Regression](https://github.com/dominikmn/one-million-posts/blob/main/modeling/log_reg.py) | Logistic Regresssion |
| [Support Vector Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/svc.py) | Support Vector Classifier |
| [Random Forest Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/random_forest.py) | Random Forest Classifier |
| [Naive Bayes Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/naive_bayes.py) | Naive Bayes Classifier |
| [LightGBM](https://github.com/dominikmn/one-million-posts/blob/main/modeling/light_gbm.py) | LightGBM algorithm not considered for further modeling |

## Data analysis
The notebooks are currently found in the main folder of this repo.
They are pushed as `.py` files. See [Setup - Notebooks](#setup---notebooks).

