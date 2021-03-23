#!/bin/bash

set -e
set -x

mkdir -p data/
mkdir -p data/raw
mkdir -p data/processed

# Download everything in raw data directory
cd data/raw/

# Download and Setup SNLI
SNLI_DATA_URL=https://nlp.stanford.edu/projects/snli/snli_1.0.zip
wget $SNLI_DATA_URL
unzip $(basename $SNLI_DATA_URL)
rm snli_1.0.zip

