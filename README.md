# One Million Posts

Natural language processing project based on the [one-million-posts dataset](https://ofai.github.io/million-post-corpus/).



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
| | Zero Shot Classifier |
| | Support Vector classifier |
| | RandomForest Classifier |
| [Naive Bayes](https://github.com/dominikmn/one-million-posts/blob/main/modeling/naive_bayes.py) | Naive Bayes classifier |

## Data analysis
The notebooks are currently found in the main folder of this repo.
They are pushed as `.py` files. See [Setup - Notebooks](#setup---notebooks).

