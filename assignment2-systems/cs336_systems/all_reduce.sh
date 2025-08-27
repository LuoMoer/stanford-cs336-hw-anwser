CUDA_VISIBLE_DEVICES="0,1,2,3"

python all_reduce.py \
    --device "cpu" \
    --world_size 6 \
    --data_size 1000000000