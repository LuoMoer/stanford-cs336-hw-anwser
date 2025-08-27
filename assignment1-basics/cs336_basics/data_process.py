from Tokenizer import *
import argparse 
import json
import pickle
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="data_process_info")
    parser.add_argument("--tokenizer_path", type=str, help="Path of tokenizer vocab and merges.")
    parser.add_argument("--data_path",  type=str, help="Path of data to process.")
    parser.add_argument("--output_path", type=str, help="Path of output processed data.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    VOCAB_PATH = os.path.join(args.tokenizer_path, "vocab.pkl")
    MERGES_PATH = os.path.join(args.tokenizer_path, "merges.pkl")
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer(vocab, merges, special_tokens)

    with open(DATA_PATH, "r") as f:
        data = f.read()
    
    data_token_ids = tokenizer.encode(data)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    train_end = int(len(data_token_ids)*0.9)

    TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "train.npy")    
    VALID_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "valid.npy")    
    np.save(TRAIN_OUTPUT_PATH, data_token_ids[:train_end])
    np.save(VALID_OUTPUT_PATH, data_token_ids[train_end:])



if __name__ == "__main__":
    main()