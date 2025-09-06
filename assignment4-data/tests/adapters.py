from __future__ import annotations

import os
from typing import Any
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import regex as re
import nltk
import unicodedata
import string
import mmh3
import random


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    html_string = html_bytes.decode(encoding=encoding)
    return extract_plain_text(html_string)


def run_identify_language(text: str) -> tuple[Any, float]:
    MODEL_PATH = "./classifiers/lid.176.bin"
    model = fasttext.load_model(MODEL_PATH)
    label, logit = model.predict(text.replace('\n',' '))
    label = label[0].replace("__label__","")
    return (label, logit[0])


def run_mask_emails(text: str) -> tuple[str, int]:
    pattern = r'(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'
    return re.subn(pattern, "|||EMAIL_ADDRESS|||", text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r'\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    return re.subn(pattern, "|||PHONE_NUMBER|||", text)


def run_mask_ips(text: str) -> tuple[str, int]:
    pattern = r'((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)'
    return re.subn(pattern, "|||IP_ADDRESS|||", text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    MODEL_PATH = "./classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
    model = fasttext.load_model(MODEL_PATH)
    prediction, logit = model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__nsfw":
        prediction = "nsfw"
    else:
        prediction = "non-nsfw"
    return (prediction, logit)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    MODEL_PATH = "./classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    model = fasttext.load_model(MODEL_PATH)
    prediction, logit = model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__toxic":
        prediction = "toxic"
    else:
        prediction = "non-toxic"
    return (prediction, logit)


def run_classify_quality(text: str) -> tuple[Any, float]:
    MODEL_PATH = "./classifiers/quality.bin"
    text = text.replace('\n',' ')
    model = fasttext.load_model(MODEL_PATH)
    prediction, logit = model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__good":
        prediction = "wiki"
    else:
        prediction = "cc"
    return (prediction, logit)


def run_gopher_quality_filter(text: str) -> bool:
    # • Contain less than 50 or more than 100,000 words.
    # • Have a mean word length outside the range of 3 to 10 characters.
    # • Have more than 30% of lines ending with an ellipsis (“...”).
    # • Contain less than 80% of words with at least one alphabetic character.
    word_list = nltk.tokenize.word_tokenize(text)
    word_num = len(word_list)
    if word_num < 50 or word_num > 100000:
        return False
    word_avg_len = sum([len(word) for word in word_list])/word_num
    if word_avg_len < 3 or word_avg_len > 10:
        return False
    ellipsis_num = sum([ (1 if re.match(r'.*\.\.\.$', line) else 0) for line in text.split('\n')]) / len(text.split('\n'))
    if ellipsis_num > 0.3:
        return False
    alphabetic_word = sum([(1 if re.match(r'[A-Za-z]', word) else 0) for word in word_list])/len(word_list)
    if alphabetic_word < 0.8:
        return False
    
    return True


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    deduplication_line = {}
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for file in input_files:
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            hs = hash(line)
            deduplication_line[hs] = deduplication_line.get(hs, 0) + 1

    for file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(file))
        of = open(output_file, "w")
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            hs = hash(line)
            if deduplication_line[hs]==1:
                of.write(line)
        of.close()
        
    
def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    band_hashes = num_hashes//num_bands
    seeds = random.sample(range(1, num_hashes*100), num_hashes)
    minhash = [] # (minhash_list, input_path)
    doc = {}
    for input_path in input_files:
        with open(input_path, "r") as f:
            document = f.read()
        document = unicodedata.normalize("NFD", document)
        document = ''.join(
            char for char in document 
            if unicodedata.category(char) != 'Mn'
        )
        document = document.lower()
        document = document.translate(str.maketrans('', '', string.punctuation))
        document = re.sub(r'\s+', ' ', document).strip()
        document = nltk.word_tokenize(document)
        document_ngrams = [str(document[i:i+ngrams]) for i in range(len(document)-ngrams)]
        doc[input_path] = set(document_ngrams)
        # num_hashs
        minhash.append((tuple([min([mmh3.hash(x, seed) for x in document_ngrams]) for seed in seeds]), input_path))
    
    fa = {input_path:input_path for input_path in input_files}
    def _find(x):
       if x!=fa[x]: fa[x]=_find(fa[x])
       return fa[x]
    
    def _union(x, y):
        fa[x] = _find(fa[y])
    
    def _jaccard(x, y):
        docx = doc[x]
        docy = doc[y]
        if not docx and not docy: return 1.0
        if not docx or not docy: return 0.0
        return len(docx & docy) / len(docx | docy)
    
    candidate_pairs = [set() for i in range(num_bands)]
    for x_list, x in minhash:
        for y_list, y in minhash:
            for i in range(num_bands):
                start = i*band_hashes
                end = start+band_hashes
                if x_list[start:end]==y_list[start:end]:
                    candidate_pairs[i].add(tuple(sorted((x,y))))
    
    candidate = {}
    for i in range(num_bands):
        for x, y in candidate_pairs[i]:
            if _find(x)==_find(y): continue
            if _jaccard(x, y) >= jaccard_threshold: _union(x,y)
                
    for x in input_files:
        candidate[_find(x)] = candidate.get(_find(x), []) + [x]

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    for input_path in candidate.keys():
        output_path = os.path.join(output_directory, os.path.basename(input_path))
        with open(input_path, 'r') as rf, open(output_path, 'w') as of:
            of.write(rf.read())