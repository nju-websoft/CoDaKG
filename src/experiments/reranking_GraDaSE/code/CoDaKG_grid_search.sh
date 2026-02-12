#!/bin/bash

set -e

gpu_id=1
lrs=(1e-5 5e-5 1e-4)
bss=(128 256)

target_metric="ndcg_cut_10"
# =================

summary_file="logs/grid_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "LR, BatchSize, LogFile, Result_Line" > "$summary_file"

echo "Start Grid Search..."
echo "Results: $summary_file"

for lr in "${lrs[@]}"; do
    for bs in "${bss[@]}"; do
        start_time=$(date +%s)
        log_file="logs/run_lr${lr}_bs${bs}_$(date +%Y%m%d_%H%M%S).log"
        
        echo "------------------------------------------------"
        echo "Running: LR=$lr, BatchSize=$bs"
        echo "Log file: $log_file"

        cmd="CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
            --dataset \"CoDaKG_tags_annotators\" \
            --len-seq 80 \
            --num-seqs 10 \
            --dropout 0.1 \
            --beta 0.1 \
            --temperature 0.2 \
            --lr $lr \
            --epoch 100 \
            --feats-type 0 \
            --num-gnns 1 \
            --batch-size $bs \
            --patience 5 \
            --num-layers 1 \
            --weight-decay 1e-4 \
            --eval-steps 200 \
            --num-heads 4 \
            --top-k 20 \
            --repeat 1"

        eval "$cmd" > "$log_file" 2>&1

        result_line=$(grep "$target_metric" "$log_file" | tail -n 1)
        
        if [ -z "$result_line" ]; then
            result_line="Error or not found"
        fi

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Completed. Result: $result_line   Time: ${duration} s"
        
        echo "$lr, $bs, $log_file, $result_line" >> "$summary_file"
        
    done
done

echo "================================================"
echo "Grid Search complete!"
echo "Best results (based on text extraction):"
cat "$summary_file"