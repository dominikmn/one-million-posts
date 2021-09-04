SHELL := /bin/bash
.PHONY: setup
setup:
		pyenv install -s 3.8.5
		pyenv local 3.8.5
		( \
			python -m venv .venv;\
			source .venv/bin/activate; \
			pip install --upgrade pip; \
			python -m pip install -r requirements.txt; \
			python -m spacy download de; \
		)
		#.venv/bin/python -m pip install --upgrade pip
		#.venv/bin/python -m pip install -r requirements.txt
		#.venv/bin/python -m spacy download de
		mkdir -p cache
		mkdir -p data
		curl http://lager.cs.uni-duesseldorf.de/NLP/IWNLP/IWNLP.Lemmatizer_20181001.zip -o ./data/lemma.zip
		unzip ./data/lemma -d ./data/

.PHONY: embeddings
embeddings:
		mkdir -p embeddings
		curl https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt -o ./embeddings/glove_vectors.txt
		curl https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt -o ./embeddings/word2vec_vectors.txt

.PHONY: dashboard
dashboard:
		pyenv local 3.8.5
		python -m venv .venv_dashboard
		.venv_dashboard/bin/python -m pip install --upgrade pip
		.venv_dashboard/bin/python -m pip install -r requirements_dashboard.txt

.PHONY: backend
backend:
		pyenv local 3.8.5
		python -m venv .venv_backend
		.venv_backend/bin/python -m pip install --upgrade pip
		.venv_backend/bin/python -m pip install -r requirements_backend.txt
