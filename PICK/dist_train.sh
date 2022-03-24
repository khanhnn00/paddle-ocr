#!/bin/bash
python3.8 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=9934 \
train.py -c config.json -d 1 --local_world_size 1 
