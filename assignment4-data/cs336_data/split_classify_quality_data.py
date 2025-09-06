from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType, WarcRecord
import gzip
from typing import Any
import os
import fasttext
from tqdm import tqdm
import regex as re
import nltk
import random
import pickle

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    try:
        html_string = html_bytes.decode(encoding=encoding)
    except:
        html_string = ""
    return extract_plain_text(html_string)


def identify_language(text: str) -> tuple[Any, float]:
    return language_model.predict(text)


def classify_nsfw(text: str):
    prediction, logit = nsfw_model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__nsfw":
        return True
    return False


def classify_toxic_speech(text: str):
    prediction, logit = hs_model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__toxic":
        return True
    return False


def gopher_quality_filter(text: str) -> bool:
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


GOOD_DATA_PATH = "./data/WKP/enwiki.warc.gz" 
BAD_DATA_PATH = "./data/CC/example.warc.gz"

LANGUAGE_MODEL_PATH = "./classifiers/lid.176.bin"
language_model = fasttext.load_model(LANGUAGE_MODEL_PATH)

NSFW_MODEL_PATH = "./classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)

HS_MODEL_PATH = "./classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
hs_model = fasttext.load_model(HS_MODEL_PATH)

print("INFO: start handle good data...")

good_data = []
with gzip.open(GOOD_DATA_PATH, "rb") as f:
    for record in ArchiveIterator(f):
        text = extract_text_from_html_bytes(record.reader.read()).replace('\n',' ')
        
        labels, logits = identify_language(text)
        if labels[0] != "__label__en" or text=="": continue
        if classify_nsfw(text) or classify_toxic_speech(text): continue
        if not gopher_quality_filter(text): continue
        good_data.append("__label__good " + text)

print("DONE: handle good data.")

print("INFO: start handle bad data...")

bad_data = []
with gzip.open(BAD_DATA_PATH, "rb") as f:
    for record in ArchiveIterator(f):
        text = extract_text_from_html_bytes(record.reader.read()).replace('\n',' ')
        
        labels, logits = identify_language(text)
        if labels[0] != "__label__en" or text=="": continue
        if classify_nsfw(text) or classify_toxic_speech(text): continue
        if not gopher_quality_filter(text): continue
        bad_data.append("__label__bad " + text)

print("DONE: handle bad data.")

print(f"INFO: good_data_size[{len(good_data)}] bad_data_size[{len(bad_data)}]")
data_size = len(good_data)+len(bad_data)

print(f"INFO: shuffle all data...")
data = good_data + bad_data
random.shuffle(data)

SAVE_TRAIN_DATA_PATH = "./data/classify_quality.train"
SAVE_VALID_DATA_PATH = "./data/classify_quality.valid"

train_data_size = int(data_size*0.9)
print(f"INFO: train_data_size[{train_data_size}] valid_data_size[{data_size-train_data_size}]")
print(f"INFO: save all data...")

with open(SAVE_TRAIN_DATA_PATH, "w") as f:
    for text in data[:train_data_size]:
        f.write(text+'\n')

with open(SAVE_VALID_DATA_PATH, "w") as f:
    for text in data[train_data_size:]:
        f.write(text+'\n')

print(f"DONE: save all data.")


