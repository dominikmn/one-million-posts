.PHONY: setup
setup:
		pyenv local 3.8.5
		python -m venv .venv
		.venv/bin/python -m pip install --upgrade pip
		.venv/bin/python -m pip install -r requirements.txt
		mkdir -p cache
		mkdir -p data
		chmod +x setup.sh
		./setup.sh
.PHONY: embeddings
embeddings:
		mkdir -p embeddings
		curl https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt -o ./embeddings/glove_vectors.txt
		curl https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt -o ./embeddings/word2vec_vectors.txt
