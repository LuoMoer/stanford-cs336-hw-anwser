from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
from einops import rearrange, einsum
from tqdm import tqdm

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [special_token.encode("utf-8") for special_token in self.special_tokens]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.token_to_id = {}
        for id, token in vocab.items():
            self.token_to_id[token]=id
        
        # Add special_token to vocab if not already present
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
    
    def encode(self, text: str) -> list[int]:
        # str -> token_dis
        import regex as re
        token_ids = []
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))

        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                token_ids.extend(self._get_token_ids(part))
        
        return token_ids
    
    def decode(self, ids: list[int]) -> str:
        # token_ids -> str
        bytes_seq = b"".join([self.vocab[id] for id in ids])
        return bytes_seq.decode("utf-8", errors="replace")
            
    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def _get_token_ids(self, text: str) -> list[int]:
        # print('!'*50+"DEBUG get_token_ids"+'!'*50)
        # print([text])
        import regex as re
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        iter = re.finditer(PAT, text)
        token_ids = []
        for item in iter:
            token_seq = item[0].encode("utf-8")
            token = self._merge(token_seq)
            token_ids.extend([self.token_to_id[b] for b in token])
        return token_ids
    
    def _merge(self, token_seq: bytes):
        token = [bytes([b]) for b in token_seq]
        def get_pairs(word):
            pairs = set()
            prev_b = word[0]
            for b in word[1:]:
                pairs.add((prev_b, b))
                prev_b = b
            return pairs
        pairs = get_pairs(token)

        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key = lambda pair:self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            a,b=bigram
            new_token=[]
            i=0
            while i < len(token):
                try:
                    j=token.index(a,i)
                except ValueError:
                    new_token.extend(token[i:])
                    break
                else:
                    new_token.extend(token[i:j])
                    i=j
                if token[i]==a and i+1<len(token) and token[i+1]==b:
                    new_token.append(a+b)
                    i+=2
                else:
                    new_token.append(token[i])
                    i+=1
            token = new_token
            if len(token) == 1:
                break
            else:
                pairs = get_pairs(token)
        return token

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # initialize vocab
    vocab = {}
    current_vocab_size = 0
    for special_token in special_tokens:
        vocab[current_vocab_size] = special_token.encode("utf-8")
        current_vocab_size+=1
    for i in range(256):
        vocab[current_vocab_size] = bytes([i])
        current_vocab_size+=1
    
    
    # pretokenization
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "r", encoding="utf-8") as f:
        texts = f.read()
        chunks = re.split("|".join(map(re.escape, special_tokens)), texts)
    
    word_num = {}
    words = {}

    for texts in chunks:
        iter = re.finditer(PAT, texts)
        for i in iter:
            word_num[i[0].encode("utf-8")] = word_num.get(i[0].encode("utf-8"), 0) + 1
            words[i[0].encode("utf-8")] = []
            for b in i[0].encode("utf-8"):
                words[i[0].encode("utf-8")].append(bytes([b]))
    
    # merge
    merges = []
    from tqdm import tqdm

    assert vocab_size >= current_vocab_size
    
    pairs = {}
    for word, word_bytes in words.items():
        for i in range(len(word_bytes)-1):
            pairs[(word_bytes[i],word_bytes[i+1])] = pairs.get((word_bytes[i],word_bytes[i+1]), 0) + word_num[word] 

    for loop in tqdm(range(vocab_size-current_vocab_size)):
        if len(pairs) == 0:
            print(f"Vocab_size too big! The max Vocab_size is [{current_vocab_size}].")
            break
        
        max_count = max(pairs.values())
        candidates = [k for k, v in pairs.items() if v == max_count]
        best_pair = max(candidates)

        merges.append(best_pair)
        vocab[current_vocab_size] = merges[-1][0]+merges[-1][1]
        current_vocab_size+=1
        
        for word in words.keys():
            new_word_bytes = []
            i = 0
            a = merges[-1][0]
            b = merges[-1][1]
            new_word = a+b
            if new_word not in word:
                continue
            while i < len(words[word]):
                if i+1 < len(words[word]) and words[word][i]==a and words[word][i+1]==b:
                    new_word_bytes.append(new_word)
                    num = word_num[word]
                    if len(new_word_bytes) > 1:
                        # 前面还有至少一个元素 new_word_bytes[-2]
                        left_word = new_word_bytes[-2]
                        pairs[(left_word,a)]-=num
                        if pairs[(left_word,a)] == 0:
                            del pairs[(left_word,a)]
                        pairs[(left_word,new_word)]=pairs.get((left_word,new_word),0)+num
                    if i+2 < len(words[word]):
                        # 后面至少还有一个元素
                        right_word = words[word][i+2]
                        pairs[(b,right_word)]-=num
                        if pairs[(b,right_word)] == 0:
                            del pairs[(b,right_word)]
                        pairs[(new_word,right_word)]=pairs.get((new_word,right_word),0)+num
                    i+=2
                else:
                    new_word_bytes.append(words[word][i])
                    i+=1
            words[word] = new_word_bytes
        del pairs[(merges[-1][0],merges[-1][1])]

    return vocab, merges