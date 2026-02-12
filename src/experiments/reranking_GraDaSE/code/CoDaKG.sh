#!/bin/bash
# 创建新进程组（可选，通常 bash 已经是组长）
set -m

# 捕获信号并清理
cleanup() {
    echo "Caught signal, killing all children..."
    kill 0  # kill 0 表示杀死当前进程组中的所有进程
    exit 1
}
trap cleanup SIGINT SIGTERM

gpu_id=0

start_time=$(date +%s)

cmd="CUDA_VISIBLE_DEVICES=$gpu_id python run.py --dataset \"CoDaKG_tags_annotators\" --len-seq 80 --num-seqs 10 --dropout 0.1 --beta 0.1 --temperature 0.2 --lr 1e-4 --epoch 100 --feats-type 0 --num-gnns 1 --batch-size 256 --patience 5 --num-layers 1 --weight-decay 1e-4 --eval-steps 200 --num-heads 4 --top-k 20 --repeat 1"

echo "Running command: $cmd"

log_file="logs/CoDaKG_tags_annotators_$(date +%Y%m%d_%H%M%S).txt"
eval "$cmd" > "$log_file" 2>&1

# 计算并显示任务耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Task CoDaKG_tags_annotators complete, time: ${duration} s. Log file: $log_file"
