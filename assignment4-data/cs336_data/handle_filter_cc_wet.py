import concurrent.futures
import os
from tqdm import tqdm
import pathlib
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
import json


def identify_language(text: str) -> tuple[Any, float]:
    prediction, logit = language_model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__en" and logit > 0.8:
        return True
    return False


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


def run_classify_quality(text: str) -> tuple[Any, float]:
    text = text.replace('\n',' ')
    prediction, logit = quality_model.predict(text)
    prediction = prediction[0]
    logit = logit[0]
    if prediction == "__label__good":
        prediction = "wiki"
    else:
        prediction = "cc"
    if prediction == "wiki":
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


def mask_pii(text: str)->str:
    pattern = r'(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))'
    text, _ = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)

    pattern = r'\(?([0-9]{3})\)?[-.\s]([0-9]{3})[-.\s]([0-9]{4})'
    text, _ = re.subn(pattern, "|||PHONE_NUMBER|||", text)

    pattern = r'((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)'
    text, _ = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return text


LANGUAGE_MODEL_PATH = "./classifiers/lid.176.bin"
language_model = fasttext.load_model(LANGUAGE_MODEL_PATH)

NSFW_MODEL_PATH = "./classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)

HS_MODEL_PATH = "./classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
hs_model = fasttext.load_model(HS_MODEL_PATH)

QUALITY_MODEL_PATH = "./classifiers/quality.bin"
quality_model = fasttext.load_model(QUALITY_MODEL_PATH)

INPUT_DIRECTORY = "./data/cc_wet/wet"
OUTPUT_DIRECTORY = "./data/cc_wet/wet_filtered"

def process_single_wet_file(input_path: str, output_path: str):
    documents = []
    with gzip.open(input_path, "rb") as f:
        for record in ArchiveIterator(f):
            document = record.reader.read()
            encoding = detect_encoding(document)
            try: document = document.decode(encoding=encoding)
            except: continue
            if "Software-Info" in document:
                continue
            doc_test = document.replace('\n','')
            if not gopher_quality_filter(doc_test): continue
            if not identify_language(doc_test): continue
            if classify_nsfw(doc_test) or classify_toxic_speech(doc_test): continue
            if not run_classify_quality(doc_test): continue
            document = mask_pii(document)
            documents.append(document)
    with open(output_path, "w") as f:
        json.dump(documents, f, ensure_ascii=False)


num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = [os.path.join(INPUT_DIRECTORY, file_name) for file_name in os.listdir(INPUT_DIRECTORY)]

futures = []
for wet_filepath in wet_filepaths:
    wet_filename = str(pathlib.Path(wet_filepath).name) + ".json"
    future = executor.submit(
        process_single_wet_file,
        wet_filepath,
        os.path.join(OUTPUT_DIRECTORY, wet_filename)
    )
    futures.append(future)

for future in tqdm(
    concurrent.futures.as_completed(futures),
    total = len(wet_filepaths)
):
    continue
