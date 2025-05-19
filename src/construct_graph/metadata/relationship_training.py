import os
import math
import logging
import itertools
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import ydf  # Yggdrasil Decision Forests

# ----- Configuration -----
STELLA_MODEL_DIR        = "stella_en_400M_v5"
GBDT_MODEL_DIR          = "gbdt"
DATASET_CSV_PATH        = "datasets_acordar.csv"
RELATIONSHIPS_CSV_PATH  = "relationships.csv"  # NEW: your labeled pairs
EMBEDDINGS_OUT          = "dataset_acordar_embeddings.npy"
IDS_OUT                 = "dataset_acordar_ids.npy"
# -------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Step 1: generate or load embeddings (unchanged)…
datasets_df   = pd.read_csv(DATASET_CSV_PATH, encoding="utf-8", na_filter=False)
dataset_ids   = datasets_df['dataset_id'].tolist()
titles        = datasets_df['title'].tolist()
descriptions  = datasets_df['description'].tolist()
N             = len(dataset_ids)

texts = ["[CLS]" + t.strip() + "[SEP]" + d.strip() for t,d in zip(titles, descriptions)]
model = SentenceTransformer(STELLA_MODEL_DIR, trust_remote_code=True).to(device)

EMBED_BATCH_SIZE = 128   # Batch size for embedding computation (adjust based on GPU memory)

embeddings_list = []
for i in range(0, N, EMBED_BATCH_SIZE):
    batch = texts[i : i + EMBED_BATCH_SIZE]
    embeds = model.encode(batch, batch_size=len(batch), show_progress_bar=False)
    embeddings_list.append(embeds)
embeddings_matrix = np.vstack(embeddings_list)
np.save(EMBEDDINGS_OUT, embeddings_matrix)
np.save(IDS_OUT, np.array(dataset_ids))

id_to_index = {ds_id: idx for idx, ds_id in enumerate(dataset_ids)}

# Step 2: Load labeled relationships, build features, train & save GBDT
logging.info("Loading labeled pairs for training...")
rels_df = pd.read_csv(RELATIONSHIPS_CSV_PATH, encoding="utf-8", na_filter=False)
# Map to embedding indices
idx1 = rels_df['dataset_id1'].map(id_to_index)
idx2 = rels_df['dataset_id2'].map(id_to_index)

logging.info("Building feature matrix for training...")
emb1 = embeddings_matrix[idx1]
emb2 = embeddings_matrix[idx2]
features = np.hstack((emb1, emb2))
feature_cols = [str(i) for i in range(features.shape[1])]

train_df = pd.DataFrame(features, columns=feature_cols)
train_df['relationship'] = rels_df['relationship']  # your binary (0/1) or multiclass labels

logging.info("Training GBDT model on pairwise features...")
learner    = ydf.GradientBoostedTreesLearner(label="relationship")  # :contentReference[oaicite:0]{index=0}
gbdt_model = learner.train(train_df)

logging.info(f"Saving trained GBDT model to '{GBDT_MODEL_DIR}'...")
gbdt_model.save(GBDT_MODEL_DIR)  # :contentReference[oaicite:1]{index=1}
logging.info("Model training and save complete.")
