# One Million Posts

Natural language processing project based on the [one-million-posts dataset](https://ofai.github.io/million-post-corpus/).

More than 3.000 user comments are written each day on www.derstandard.at. Moderators need to review these comments regarding several aspects like inappropriate language, discriminating content, off-topic comments, questions that need to be answered, and more.	

We provide machine learning models that detect potentially problematic comments to ease the moderators' daily work.

## Setup
1. Install [pyenv](https://github.com/pyenv/pyenv).
2. Install python 3.8.5 via `pyenv install 3.8.5`
3. Run `make setup`. 

For further instructions on how to run our code see [SETUP.md](SETUP.md).

## Presentations
The presentations are found in `./presentations/`
| Presentation file | Description |
|-|-|
| [OneMillionPosts-GraduationEvent.pdf](https://github.com/dominikmn/one-million-posts/blob/main/presentations/OneMillionPosts-GraduationEvent.pdf) | Presentation of the graduation event from April 28, 2021 |
| [OneMillionPosts-Midterm.pdf](https://github.com/dominikmn/one-million-posts/blob/main/presentations/OneMillionPosts-Midterm.pdf) | Midterm presentation of the project from April 12, 2021 |
| [OneMillionPosts-AnnotationComposition.pdf](https://github.com/dominikmn/one-million-posts/blob/main/presentations/OneMillionPosts-AnnotationComposition.pdf) | EDA concerning ticket [#24][i24], [#25][i25] |

[i24]: https://github.com/dominikmn/one-million-posts/issues/24
[i25]: https://github.com/dominikmn/one-million-posts/issues/25

## Modeling
The models' code is found in  `./modeling/` in this repo.
They are pushed as `.py` files. See [SETUP.md](SETUP.md).
| Model | Description |
|-|-|
| [gbert Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/gbert_classifier.py) | [German BERT base](https://huggingface.co/deepset/gbert-base) | 
| [Zero Shot Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/modeling_zero_shot.py) | [xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) |
| [XGBoost](https://github.com/dominikmn/one-million-posts/blob/main/modeling/xg_boost.py) | XGBoost |
| [Logistic Regression](https://github.com/dominikmn/one-million-posts/blob/main/modeling/log_reg.py) | Logistic Regresssion |
| [Support Vector Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/svc.py) | Support Vector Classifier |
| [Random Forest Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/random_forest.py) | Random Forest Classifier |
| [Naive Bayes Classifier](https://github.com/dominikmn/one-million-posts/blob/main/modeling/naive_bayes.py) | Naive Bayes Classifier |
| [LightGBM](https://github.com/dominikmn/one-million-posts/blob/main/modeling/light_gbm.py) | LightGBM algorithm not considered for further modeling |

## Data analysis
The notebooks are found in `./notebooks/`.
They are pushed as `.py` files. See [SETUP.md](SETUP.md).

