from Tokenizer import *
import argparse 
import json
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='tarin_info')
    parser.add_argument('--input_path', type=str, help='The input path for tokenizer training.') 
    parser.add_argument('--output_path', type=str, help='The save path for tokenizer.') 
    parser.add_argument('--vocab_size', type=int, help='Vocab size for tokenizer.') 
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(args.input_path, args.vocab_size, special_tokens)
    vocab_output_path = args.output_path + "/vocab.pkl"
    merges_output_path = args.output_path + "/merges.pkl"

    with open(vocab_output_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_output_path, "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    main()