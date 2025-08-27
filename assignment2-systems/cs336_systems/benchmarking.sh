CUDA_VISIBLE_DEVICES="0"

uv run python benchmarking.py \
    --model_config "./model_configs/small.json" \
    --warm_up 5 \
    --active 10