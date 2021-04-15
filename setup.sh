#!/bin/sh

python -m spacy download de
wget -O ./data/lemma.zip http://lager.cs.uni-duesseldorf.de/NLP/IWNLP/IWNLP.Lemmatizer_20181001.zip

cd data
unzip lemma
cd ..