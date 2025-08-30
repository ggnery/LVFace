#!/bin/bash

set -e

mkdir -p data/lfw 

# LFW
cd data/lfw
curl -L -o lfw-dataset.zip https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset
unzip lfw-dataset.zip 
rm lfw-dataset.zip