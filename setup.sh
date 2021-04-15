#!/bin/sh

source .venv/bin/activate

python -m spacy download de
curl http://lager.cs.uni-duesseldorf.de/NLP/IWNLP/IWNLP.Lemmatizer_20181001.zip -o ./data/lemma.zip

cd data
unzip lemma
cd ..