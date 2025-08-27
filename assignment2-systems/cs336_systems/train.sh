export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ./train.py \
    --config_path "./config_set/config_single.json" \
    --train_data_path "./processed_data/TinyStories/train.npy" \
    --valid_data_path "./processed_data/TinyStories/valid.npy"
