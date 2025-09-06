from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType, WarcRecord
import gzip
import fasttext
from typing import Any


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    try:
        html_string = html_bytes.decode(encoding=encoding)
    except:
        html_string = ""
    return extract_plain_text(html_string)


def run_identify_language(text: str) -> tuple[Any, float]:
    return model.predict(text)


data_path = "../data/CC/example.warc.wet.gz"
Test_siz = 100

texts = []
model = fasttext.load_model("../classifiers/lid.176.bin")

with gzip.open(data_path, "rb") as f:
    for record in ArchiveIterator(f):
        text = run_extract_text_from_html_bytes(record.reader.read()).replace("\n"," ")
        
        labels, logits = run_identify_language(text)
        if labels[0] != "__label__en" or text=="":
            continue
        
        print(f"ID: {len(texts)}\tLabel: {labels[0]}\tLogit: {logits[0]}\n{text}\n")
        
        texts.append((text, labels[0]))
        if len(texts)==Test_siz:
            break