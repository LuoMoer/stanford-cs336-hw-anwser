from utils import *
from modeling import *
from Tokenizer import *
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
import swanlab
import pickle

def get_args():
    parser = argparse.ArgumentParser(description="generate_info")
    parser.add_argument("--tokenizer_path", type=str, help="Path of tokenizer vocab and merges.")
    parser.add_argument("--checkpoint_path", type=str, help="Path of checkpoint.")
    parser.add_argument("--config_path", type=str, help="Path of config.")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    CONFIG_PATH = "/home/swluo/data/CS336/assignment1-basics/cs336_basics/config.json"
    CHECKPOINT_PATH = "/home/swluo/data/CS336/assignment1-basics/cs336_basics/checkpoint/TinyStroy_0801/checkpoint-2000.pth"
    VOCAB_PATH = os.path.join(args.tokenizer_path, "vocab.pkl")
    MERGES_PATH = os.path.join(args.tokenizer_path, "merges.pkl")


    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    model_config = config["model_config"]
    optimizer_config = config["optimizer_config"]

    model = transformer_lm(**model_config)
    optimizer = AdamW(
        model.parameters(), learning_rate_schedule, **optimizer_config)
    
    load_checkpoint(CHECKPOINT_PATH, model, optimizer)

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    
    prompt = "Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one."

    print(model.generate(prompt, tokenizer, max_seq_len = 256, temperature = 0.6, top_p = 0.6))

if __name__ == "__main__":
    main()