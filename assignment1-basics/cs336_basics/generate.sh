export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0"

python cs336_basics/generate.py \
    --tokenizer_path "cs336_basics/tokenizer" \
    --checkpoint_path "cs336_basics/checkpoint/TinyStory_0801/checkpoint-2000.pth" \
    --config_path "cs336_basics/config.json" \
