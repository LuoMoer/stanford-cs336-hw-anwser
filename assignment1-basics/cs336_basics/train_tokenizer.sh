export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0"

uv run python cs336_basics/train_tokenizer.py \
    --input_path "data/TinyStoriesV2-GPT4-valid.txt" \
    --output_path "cs336_basics/tokenizer" \
    --vocab_size 10000 \
