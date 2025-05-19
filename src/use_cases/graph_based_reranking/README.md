## 1. Descriptions
The repository is organised as follows:

- run.py: reranking of NTCIR.
- model.py: implementation of HINormer(ref: https://github.com/Ffffffffire/HINormer).
- utils/: contains tool functions.


## 2. Requirements

- Python==3.9.0
- Pytorch==1.12.0
- Networkx==2.8.4
- numpy==1.22.3
- dgl==0.9.0
- scikit-learn==1.1.1
- scipy==1.7.3

## 3. Running experiments

We train our model using NVIDIA GeForce RTX 4090 GPU with CUDA 12.2.

For evaluation of NTCIR:
```
python run.py --dataset ntcir_metadata_content --len-seq 80 --epoch 50 --patience 6 --num-gnns 1 --num-layers 1 --num-heads 4 --lr 1e-5 --dropout 0.1 --beta 0.5 --temperature 2 --feats-type 0 --batch-size 1 --top-k 10 --mode bm25
```
