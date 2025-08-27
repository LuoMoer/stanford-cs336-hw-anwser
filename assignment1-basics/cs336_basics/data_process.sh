export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0"

uv run python cs336_basics/data_process.py \
    --tokenizer_path "cs336_basics/tokenizer" \
    --data_path "data/TinyStoriesV2-GPT4-valid.txt" \
    --output_path "cs336_basics/processed_data/TinyStories" \
