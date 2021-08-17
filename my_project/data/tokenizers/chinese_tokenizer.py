# -*- coding: utf-8 -*-

import os
from typing import List
from overrides import overrides

import jieba
import jieba.posseg as poss

from allennlp.data.tokenizers import Tokenizer, Token


@Tokenizer.register('jieba')
class JIEBASplitter(Tokenizer):
    """
    用jieba进行分词,可以定义用户词典和停用词词典.
    """

    def __init__(self,
                 pos_tags: bool = False,
                 only_tokens: bool = True,
                 user_dict: str = None,
                 stop_words_path: str = None) -> None:

        jieba.enable_parallel(4)
        self._pos_tags = pos_tags  # 是否标注词性。

        if user_dict and os.path.exists(user_dict):
            jieba.load_userdict(user_dict)
        self._only_tokens = only_tokens  # 最终是否只保留字符，去掉词性等属性

        self._stop_words = None  # 停用词
        if stop_words_path:
            self._stop_words = set()
            with open(stop_words_path, 'r') as f:
                for line in f:
                    word = line.strip()
                    self._stop_words.add(word)

    def _sanitize(self, tokens) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        sanitize_tokens = []
        if self._pos_tags:
            for text, pos in tokens:
                if self._stop_words and text in self._stop_words:
                    continue
                token = Token(text)
                if self._only_tokens:
                    pass
                else:
                    token = Token(token.text,
                                  token.idx,
                                  token.lemma_,
                                  pos,
                                  token.tag_,
                                  token.dep_,
                                  token.ent_type_)
                sanitize_tokens.append(token)
        else:
            for token in tokens:
                if self._stop_words and token in self._stop_words:
                    continue
                token = Token(token)
                sanitize_tokens.append(token)
        return sanitize_tokens

    @overrides
    def batch_tokenize(self, sentences: List[str]) -> List[List[Token]]:
        split_words = []
        if self._pos_tags:
            for sent in sentences:
                split_words.append(self._sanitize(tokens) for tokens in poss.cut(sent))
        else:
            for sent in sentences:
                split_words.append(self._sanitize(tokens) for tokens in jieba.cut(sent))
        return split_words

    @overrides
    def tokenize(self, sentence: str) -> List[Token]:
        if self._pos_tags:
            return self._sanitize(poss.cut(sentence))
        else:
            return self._sanitize(jieba.cut(sentence))


if __name__ == '__main__':
    splitter = JIEBASplitter()
    for token in splitter.tokenize('武汉市长江大桥'):
        print(token)
