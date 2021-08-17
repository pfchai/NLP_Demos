# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List
from overrides import overrides

# import jieba

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register('weibo2018')
class Weibo2018Reader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_sequence_length = max_sequence_length

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length]
        text_field = TextField(tokens, self.token_indexers)
        #  TODO: <16-08-21, chaipf> #  原来是text，改成tokens正常，原因待查
        fields: Dict[str, Field] = {"tokens": text_field}
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                _, sentiment, text = line.strip().split(',', 2)
                # yield self.text_to_instance(' '.join(jieba.lcut(text)), sentiment)
                yield self.text_to_instance(text, sentiment)


if __name__ == '__main__':
    reader = Weibo2018Reader()
    dataset = list(reader.read("../../../dataset/weibo2018/test.txt"))

    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))
    print(dataset[0])
