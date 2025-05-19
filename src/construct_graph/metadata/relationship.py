import os
import math
import logging
import itertools
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import ydf  # Yggdrasil Decision Forests library for the GBDT model

# ----- Configuration -----
STELLA_MODEL_DIR = "stella_en_400M_v5"   # Directory of the Stella SentenceTransformer model
GBDT_MODEL_DIR   = "gbdt"     # Directory of the saved GBDT model
DATASET_CSV_PATH = "datasets_acordar.csv"        # CSV containing columns: dataset_id, title, description
EMBEDDINGS_OUT   = "dataset_acordar_embeddings.npy"       # Optional: file to save embeddings matrix
IDS_OUT          = "dataset_acordar_ids.npy"              # Optional: file to save dataset ID list
OUTPUT_CSV_PATH  = "pairs_relationships_acordar.csv"      # Output CSV for pairwise relationships
PROGRESS_PATH    = "progress_acordar.log"                # Progress log file

EMBED_BATCH_SIZE = 128   # Batch size for embedding computation (adjust based on GPU memory)
PAIR_BATCH_SIZE  = 1028  # Number of pairs per prediction batch (given in problem statement)
# -------------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Ensure we use only one GPU (if multiple are available)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Step 1: Load dataset information and generate embeddings
logging.info("Loading dataset information...")
datasets_df = pd.read_csv(DATASET_CSV_PATH, encoding="utf-8", na_filter=False)
# Each row in datasets_df has 'dataset_id', 'title', 'description'
dataset_ids = datasets_df['dataset_id'].tolist()
titles = datasets_df['title'].tolist()
descriptions = datasets_df['description'].tolist()
N = len(dataset_ids)
logging.info(f"Loaded {N} datasets from CSV.")

# Prepare input texts for the Stella model: "[CLS] title [SEP] description"
texts = []
for title, desc in zip(titles, descriptions):
    text = "[CLS]" + title.strip() + "[SEP]" + desc.strip()
    texts.append(text)
logging.info("Prepared text for each dataset. Computing embeddings with Stella model...")

# Load the Stella SentenceTransformer model and encode texts in batches
model = SentenceTransformer(STELLA_MODEL_DIR, trust_remote_code=True).to(device)
# Encode in batches to avoid memory issues
embeddings_list = []
for i in range(0, N, EMBED_BATCH_SIZE):
    batch_texts = texts[i : i + EMBED_BATCH_SIZE]
    batch_embeds = model.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False)
    embeddings_list.append(batch_embeds)
embeddings_matrix = np.vstack(embeddings_list)
embed_dim = embeddings_matrix.shape[1]
logging.info(f"Generated embeddings for all datasets. Embedding dimension = {embed_dim}.")

# (Optional) Save embeddings and IDs to disk for reuse
np.save(EMBEDDINGS_OUT, embeddings_matrix)
np.save(IDS_OUT, np.array(dataset_ids))
logging.info(f"Saved embeddings to '{EMBEDDINGS_OUT}' and dataset IDs to '{IDS_OUT}' for future use.")

# Create a mapping from dataset_id to index in the embeddings matrix (for quick lookup)
id_to_index = {ds_id: idx for idx, ds_id in enumerate(dataset_ids)}

# Step 2: Load the pre-trained GBDT model using YDF
logging.info("Loading pre-trained GBDT model for relationship prediction...")
gbdt_model = ydf.load_model(GBDT_MODEL_DIR)  # Load the model directory&#8203;:contentReference[oaicite:4]{index=4}
logging.info("GBDT model loaded successfully.")

# Step 3: Set up output CSV and progress logging for batch processing of pairs
# Determine resume point if progress log exists
start_batch = 0
if os.path.exists(PROGRESS_PATH):
    try:
        with open(PROGRESS_PATH, "r") as pf:
            lines = pf.read().splitlines()
            if lines:
                last_completed = int(lines[-1].strip())
                start_batch = last_completed + 1  # resume from next batch
                logging.info(f"Resuming from batch {start_batch} (based on progress log).")
    except Exception as e:
        logging.warning(f"Could not read progress file, starting from batch 0. Error: {e}")

# Open the output CSV file in append mode if resuming, otherwise write mode and add header
file_mode = 'a' if start_batch > 0 and os.path.exists(OUTPUT_CSV_PATH) else 'w'
out_file = open(OUTPUT_CSV_PATH, file_mode, buffering=1)
if file_mode == 'w':
    out_file.write("dataset_id1,dataset_id2,relationship\n")  # CSV header

# Prepare for pair iteration
total_pairs = N * (N - 1) // 2
total_batches = math.ceil(total_pairs / PAIR_BATCH_SIZE)
logging.info(f"Total unique pairs: {total_pairs}. Processing in {total_batches} batches of size {PAIR_BATCH_SIZE}.")

# Initialize counters for skipping pairs (if resuming)
skip_pairs = start_batch * PAIR_BATCH_SIZE
pairs_processed = 0
current_batch = start_batch

logging.info(f"Starting pairwise relationship predictions at batch {current_batch}...")

try:
    # Iterate through all unique pairs (i < j)
    batch_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            # Count this pair in the overall sequence
            pairs_processed += 1
            # If we haven't reached the starting pair, skip it
            if pairs_processed <= skip_pairs:
                continue
            # Collect the pair for the current batch
            batch_pairs.append((dataset_ids[i], dataset_ids[j]))
            # When batch size is reached, perform prediction on this batch
            if len(batch_pairs) == PAIR_BATCH_SIZE:
                # Prepare feature matrix for this batch (concatenate embeddings)
                idx1 = [id_to_index[p[0]] for p in batch_pairs]
                idx2 = [id_to_index[p[1]] for p in batch_pairs]
                embed1 = embeddings_matrix[idx1]  # shape (batch_size, embed_dim)
                embed2 = embeddings_matrix[idx2]  # shape (batch_size, embed_dim)
                features = np.hstack((embed1, embed2))  # shape (batch_size, 2*embed_dim)
                # Create a DataFrame for the features if needed by ydf (YDF can accept numpy directly too)
                num_features = features.shape[1]
                columns = [str(i) for i in range(num_features)]
                feature_df = pd.DataFrame(features, columns=columns)
                # Predict relationships for the batch
                predictions = gbdt_model.predict(feature_df)
                # If model is binary classification, `predictions` is an array of probabilities&#8203;:contentReference[oaicite:5]{index=5}.
                # Convert to binary labels (0/1) by thresholding at 0.5
                if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
                    # Binary classification (single probability output) or regression style output
                    preds = (predictions.flatten() >= 0.5).astype(int)
                else:
                    # Multi-class classification: take the class with highest probability
                    class_indices = np.argmax(predictions, axis=1)
                    preds = class_indices.astype(int)
                # Write results to CSV for this batch
                lines = []
                for (ds1, ds2), rel in zip(batch_pairs, preds):
                    if rel != 0:
                        lines.append(f"{ds1},{ds2},{rel}\n")
                out_file.writelines(lines)
                out_file.flush()
                # Log progress and update batch index
                with open(PROGRESS_PATH, "a") as pf:
                    pf.write(f"{current_batch}\n")
                logging.info(f"Batch {current_batch} complete. Written {len(batch_pairs)} pair predictions to CSV.")
                current_batch += 1
                batch_pairs = []  # reset for next batch
    # After loop, handle any remaining pairs in the final batch (if total_pairs not exactly divisible by batch size)
    if batch_pairs:
        # Prepare features for the remaining batch
        idx1 = [id_to_index[p[0]] for p in batch_pairs]
        idx2 = [id_to_index[p[1]] for p in batch_pairs]
        embed1 = embeddings_matrix[idx1]
        embed2 = embeddings_matrix[idx2]
        features = np.hstack((embed1, embed2))
        num_features = features.shape[1]
        columns = [str(i) for i in range(num_features)]
        feature_df = pd.DataFrame(features, columns=columns)
        predictions = gbdt_model.predict(feature_df)
        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
            preds = (predictions.flatten() >= 0.5).astype(int)
        else:
            preds = np.argmax(predictions, axis=1).astype(int)
        for (ds1, ds2), rel in zip(batch_pairs, preds):
            if rel != 0:
                out_file.write(f"{ds1},{ds2},{rel}\n")
        out_file.flush()
        with open(PROGRESS_PATH, "a") as pf:
            pf.write(f"{current_batch}\n")
        logging.info(f"Final batch {current_batch} complete (size {len(batch_pairs)}).")
except Exception as e:
    logging.error(f"An error occurred during batch processing: {e}", exc_info=True)
finally:
    out_file.close()
    logging.info("Processing finished. Output saved to CSV. You can use the progress log to verify all batches were processed.")
