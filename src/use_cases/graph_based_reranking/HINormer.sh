#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run_acordar.py --dataset acordar_metadata_content --len-seq 10 --epoch 50 --patience 6 --num-gnns 1 --num-layers 1 --num-heads 4 --lr 1e-5 --dropout 0.1 --beta 0.5 --temperature 2 --feats-type 0 --batch-size 128 --eval-steps 100 --top-k 10 --repeat 1 --mode bm25 --hidden-dim 256 --fold 0
