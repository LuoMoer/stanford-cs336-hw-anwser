export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0"

python cs336_basics/train.py \
    --config_path "cs336_basics/config_set/config_lr.json" \
    --train_data_path "cs336_basics/processed_data/TinyStories/train.npy" \
    --valid_data_path "cs336_basics/processed_data/TinyStories/valid.npy"
