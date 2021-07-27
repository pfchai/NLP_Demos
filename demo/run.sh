#!/bin/bash

# train
# SST 情感分类示例
allennlp train demo/basic_stanford_sentiment_treebank.jsonnet -s log --include-package my_project
