import concurrent.futures
from typing import Any
import os
from tqdm import tqdm
import regex as re
import nltk, random, json
import unicodedata, mmh3, string


jaccard_threshold: float = 0.8

def cal_minhash(
    documents,
    seeds,
    input_path,
    num_hashes: int = 100,
    num_bands: int = 10,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8
):
    band_hashes = num_hashes//num_bands
    minhash = [] # doc: {}
    for document in documents:
        text = document
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
        document_ngrams = set(document_ngrams)
        # num_hashs
        minhash.append({
            "minhash": tuple([min([mmh3.hash(x, seed) for x in document_ngrams]) for seed in seeds]), 
            "input_path": input_path,
            "ngram": document_ngrams,
            "text": text,
        })
    return minhash


INPUT_DIRECTORY = "./data/cc_wet/wet_filtered"
OUTPUT_DIRECTORY = "./data/cc_wet/wet_deduplication"

def process_single_file(input_path: str, seeds):
    with open(input_path, "r") as f:
        documents = json.load(f)
    return cal_minhash(documents, seeds, input_path)


num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = [os.path.join(INPUT_DIRECTORY, file_name) for file_name in os.listdir(INPUT_DIRECTORY)]

futures = []
num_hashes = 100
num_bands = 10
band_hashes = num_hashes//num_bands


# cal minhashes for all documents
for wet_filepath in wet_filepaths:
    seeds = random.sample(range(1, num_hashes*100), num_hashes)
    future = executor.submit(
        process_single_file,
        wet_filepath,
        seeds
    )
    futures.append(future)

minhashes = []
for future in tqdm(
    concurrent.futures.as_completed(futures),
    total = len(wet_filepaths)
):
    minhashes += future.result()


# union duplicattion
fa = range(len(minhashes))

def _find(x):
    if x!=fa[x]: fa[x]=_find(fa[x])
    return fa[x]

def _union(x, y):
    fa[x] = _find(fa[y])

def _jaccard(x, y):
    docx = minhashes[x]["ngram"]
    docy = minhashes[y]["ngram"]
    if not docx and not docy: return 1.0
    if not docx or not docy: return 0.0
    return len(docx & docy) / len(docx | docy)

candidate_pairs = [set() for i in range(num_bands)]
for x, x_minhash in enumerate(tqdm(minhashes)):
    for y, y_minhash in enumerate(minhashes):
        for i in range(num_bands):
            start = i*band_hashes
            end = start+band_hashes
            x_list = x_minhash["minhash"]
            y_list = y_minhash["minhash"]
            if x_list[start:end]==y_list[start:end]:
                candidate_pairs[i].add(tuple(sorted((x,y))))

candidate = {}
for i in range(num_bands):
    for x, y in candidate_pairs[i]:
        if _find(x)==_find(y): continue
        if _jaccard(x, y) >= jaccard_threshold: _union(x,y)
            
for x in range(len(minhashes)):
    candidate[_find(x)] = candidate.get(_find(x), []) + [x]

if not os.path.exists(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)

sorted(candidate, key="input_path")
old_output_path = ""
save_documents = []
for x in tqdm(candidate.keys()):
    doc = minhashes[x]
    output_path = os.path.join(OUTPUT_DIRECTORY, os.path.basename(doc["input_path"]))
    if old_output_path!="" and output_path != old_output_path:
        with open(output_path, "w") as of:
            json.dump(save_documents, of)
    save_documents.append(doc["text"])
    old_output_path = output_path
    