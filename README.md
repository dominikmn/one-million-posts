# one-million-posts

Natural language processing projects based on the [one-million-posts dataset](https://ofai.github.io/million-post-corpus/).

## Setup
1. Install [pyenv](https://github.com/pyenv/pyenv).
2. Install python 3.8.5 via `pyenv install 3.8.5`
3. Run `make setup`. 

### Setup - Modeling
See [SETUP.md](SETUP.md).

### Setup - Notebooks
The notebooks are pushed as `.py` files in the _python percentage script_ format (we like meaningful diffs).  
These files have been created via the jupyter plugin [jupytext](https://github.com/mwouts/jupytext) which will automatically get installed if you execute `make setup` as part of the basic [setup](#setup).
To get the actual notebook experience open them via jupyter. But even without jupytext you can run them just like any python file via `python -m file_name.py`.

## Presentations
The presentations are found in `./presentations/`
| Presentation file | Description |
|-|-|
| [One Million Posts - Annotation composition.pdf](https://github.com/dominikmn/one-million-posts/blob/general-readme-update-midterm/presentations/One%20Million%20Posts%20-%20Annotation%20composition.pdf) | EDA concerning ticket [#24][i24], [#25][i25] |

[i24]: https://github.com/dominikmn/one-million-posts/issues/24
[i25]: https://github.com/dominikmn/one-million-posts/issues/25

## Modeling
The models' code is found in  `./modeling/` in this repo.
They are pused as `.py` files. See [Setup - Modeling](#setup-modeling)
| Model | Description |
|-|-|
| | Zero Shot Classifier |
| | Support Vector classifier |
| | RandomForest Classifier |
| [Naive Bayes](https://github.com/dominikmn/one-million-posts/blob/main/modeling/naive_bayes.py) | Naive Bayes classifier |

## Data analysis
The notebooks are currently found in the main folder of this repo.
They are pushed as `.py` files. See [Setup - Notebooks](#setup-notebooks).

