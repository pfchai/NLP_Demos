#!/bin/bash

# train
## SST 情感分类示例
CUDA_VISIBLE_DEVICES=1 allennlp train demo/basic_stanford_sentiment_treebank.jsonnet -s log -f --include-package my_project

## 中文文本分类示例
CUDA_VISIBLE_DEVICES=1 allennlp train demo/basic_chinese_classifier.jsonnet -s log -f --include-package my_project

