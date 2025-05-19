import os
import re
import json
import sys
import logging
import string
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO # Used by pandas.read_csv with truncated content

import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd

# Set multiprocessing start method
multiprocessing.set_start_method("spawn", force=True)

# ---------------- Constants ------------------
# File Processing Constants
DATA_FOLDER = "data"
CHECKPOINT_FILE_FILES = os.path.abspath("checkpoint.json")
RESULT_CSV_FILES = "keyphrase_results.csv"
DB_FILE_META_TABLE = "ntcir_datafile" # Table for file metadata
DB_RESULT_TABLE_FILES = "keyphrase_results_512_new" # Table to save results for file processing

# ACORDAR Processing Constants
CHECKPOINT_FILE_ACORDAR = os.path.abspath("checkpoint_acordar.json")
RESULT_CSV_ACORDAR = "acordar_keyphrase_results.csv"
DB_TRIPLE_TABLE = "rdf_triple_copy1" # Table for RDF triples
DB_TERM_TABLE = "rdf_term" # Table for RDF terms
DB_RESULT_TABLE_ACORDAR = "acordar_keyphrase_results" # Table to save results for ACORDAR

# Database Configuration
DB_CONFIG = {
    "host": "...",
    "port": 3306,
    "user": "...",
    "password": "...",
    "database": "..."
}

# Model Configuration
MODEL_NAME = "bloomberg/KeyBART"
GPU_LIST = [0, 1, 2, 3, 4] # List of GPU IDs to use

# Keyphrase Generation Parameters
MAX_GENERATE_LENGTH = 500
MIN_GENERATE_LENGTH = 400
MIN_KEYPHRASE_COUNT = 100 # Target number of keyphrases

# File Handling Parameters
FILE_SIZE_THRESHOLD = 500 * 1024 * 1024 # Threshold for file size truncation

# File Type Grouping (Based on original script 1)
GROUP_A = {"xlsx", "csv", "json", "xml", "xls"}
GROUP_B = {"pdf", "txt", "html", "doc", "docx"} # Treated as plain text in this script

# ---------------- Global Variables (for subprocesses) ------------------
# These are loaded once per process in init_worker
global_tokenizer = None
global_model = None
global_device = None

# ---------------- Helper Functions ------------------
def is_digit_and_punct(word):
    """
    Checks if a word consists only of digits and punctuation.
    """
    return all(ch.isdigit() or ch in string.punctuation for ch in word)

def init_worker(gpu_list):
    """
    Initializes each worker process: loads the model and tokenizer
    and assigns a GPU based on process ID.
    """
    global global_tokenizer, global_model, global_device
    try:
        proc_id = int(multiprocessing.current_process()._identity[0])
    except Exception:
        proc_id = 0 # Default to 0 if process ID cannot be determined
    gpu_id = gpu_list[(proc_id - 1) % len(gpu_list)] if gpu_list else 0 # Handle empty gpu_list
    if torch.cuda.is_available():
        global_device = torch.device(f"cuda:{gpu_id}")
        logging.info(f"Worker process {proc_id} initialized on GPU {gpu_id}")
    else:
        global_device = torch.device("cpu")
        logging.info(f"Worker process {proc_id} initialized on CPU")

    # Load model and tokenizer
    try:
        global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        global_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(global_device)
        # Suppress excessive logging from transformers
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    except Exception as e:
        logging.error(f"Worker process {proc_id} failed to load model/tokenizer: {e}")
        sys.exit(1) # Exit the worker if model loading fails

    # Set signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

def load_checkpoint(checkpoint_path):
    """
    Loads the set of processed item IDs from a checkpoint file.
    Returns an empty set if the file doesn't exist or is invalid.
    """
    processed = set()
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf8") as f:
                processed = set(json.load(f))
            logging.info(f"Loaded checkpoint from {checkpoint_path} with {len(processed)} items.")
        except Exception:
             logging.warning(f"Could not load or parse checkpoint file {checkpoint_path}. Starting fresh.")
             processed = set() # Start fresh if file is corrupted
    else:
        logging.info(f"Checkpoint file {checkpoint_path} not found. Starting fresh.")
    return processed

def save_checkpoint(processed_set, checkpoint_path):
    """
    Saves the set of processed item IDs to a checkpoint file.
    """
    try:
        with open(checkpoint_path, "w", encoding="utf8") as f:
            json.dump(list(processed_set), f)
        # logging.info(f"Saved checkpoint to {checkpoint_path} with {len(processed_set)} items.")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")

# ---------------- File Cleaning Functions (from original script 1) ------------------
def clean_text_file(content):
    """
    Cleans plain text content by removing empty lines and words
    consisting only of digits and punctuation.
    """
    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if is_digit_and_punct(line.replace(" ", "")):
            continue
        words = line.split()
        filtered_words = [w for w in words if not is_digit_and_punct(w)]
        if filtered_words:
            cleaned_lines.append(" ".join(filtered_words))
    return "\n".join(cleaned_lines)

def safe_read_csv(file_path, truncated=False):
    """
    Attempts to read a CSV file with multiple encodings and engines.
    Optionally reads only the beginning of the file if truncated=True.
    """
    attempts = [
        {"encoding": "utf8", "engine": "c", "delimiter": None},
        {"encoding": "utf8", "engine": "python", "delimiter": None},
        {"encoding": "cp1252", "engine": "c", "delimiter": None},
        {"encoding": "cp1252", "engine": "python", "delimiter": None},
    ]
    for params in attempts:
        try:
            if truncated:
                with open(file_path, "r", encoding=params["encoding"], errors="ignore") as f:
                    content = f.read(FILE_SIZE_THRESHOLD)
                # Use StringIO to treat the string content as a file
                df = pd.read_csv(StringIO(content),
                                 encoding=params["encoding"],
                                 engine=params["engine"],
                                 on_bad_lines='skip',
                                 low_memory=False)
            else:
                df = pd.read_csv(file_path,
                                 encoding=params["encoding"],
                                 engine=params["engine"],
                                 on_bad_lines='skip',
                                 low_memory=False)
            logging.debug(f"Successfully read {file_path} with encoding={params['encoding']}, engine={params['engine']}")
            return df
        except Exception as e:
            logging.debug(f"Failed to read {file_path} with encoding={params['encoding']}, engine={params['engine']}: {e}")
    logging.error(f"All attempts to read {file_path} failed.")
    return None

def safe_read_excel(file_path):
    """
    Safely reads an Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        logging.error(f"Failed to read Excel file {file_path}: {e}")
        return None

def clean_table_file(file_path, truncated=False):
    """
    Cleans content from table files (CSV, Excel) by reading and
    concatenating string representations of cells, filtering out
    digits/punctuation-only content.
    """
    ext = os.path.splitext(file_path)[1].lower()
    df = None
    if "xlsx" in ext or "xls" in ext:
        df = safe_read_excel(file_path)
    else: # Assumes CSV if not explicitly Excel
        df = safe_read_csv(file_path, truncated=truncated)

    if df is None:
        return ""

    def clean_cell(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        if is_digit_and_punct(s):
            return ""
        return s

    # Apply cleaning to each cell and flatten the DataFrame to a single string
    df_clean = df.applymap(clean_cell)
    text = " ".join(df_clean.fillna("").astype(str).values.flatten())
    return text

def clean_json_file(file_path, truncated=False):
    """
    Cleans JSON file content by recursively extracting values and
    concatenating them, filtering out digits/punctuation-only values.
    Optionally reads only the beginning of the file if truncated=True
    (may result in invalid JSON).
    """
    def extract_values(obj):
        vals = []
        if isinstance(obj, dict):
            for v in obj.values():
                vals.extend(extract_values(v))
        elif isinstance(obj, list):
            for item in obj:
                vals.extend(extract_values(item))
        elif isinstance(obj, (str, int, float)):
            s = str(obj).strip()
            if not is_digit_and_punct(s):
                vals.append(s)
        return vals

    try:
        if truncated:
            with open(file_path, "r", encoding="utf8", errors="ignore") as f:
                content = f.read(FILE_SIZE_THRESHOLD)
            # Attempt to load JSON from truncated content
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error in {file_path} (might be due to truncation): {e}")
                return "" # Return empty string if truncated JSON is invalid
            except Exception as e:
                 logging.warning(f"Error processing truncated JSON in {file_path}: {e}")
                 return ""
        else:
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
        values = extract_values(data)
        return " ".join(values)
    except FileNotFoundError:
         logging.error(f"JSON file not found: {file_path}")
         return ""
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {file_path}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error cleaning JSON file {file_path}: {e}")
        return ""


def clean_xml_file(file_path, truncated=False):
    """
    Cleans XML file content by stripping tags and filtering out
    digits/punctuation-only words. Handles .xml, .rdf.
    Optionally reads only the beginning of the file if truncated=True.
    """
    try:
        with open(file_path, "r", encoding="utf8", errors="ignore") as f:
            content = f.read(FILE_SIZE_THRESHOLD) if truncated else f.read()
    except FileNotFoundError:
         logging.error(f"XML file not found: {file_path}")
         return ""
    except Exception as e:
        logging.error(f"Error reading XML file {file_path}: {e}")
        return ""

    # Remove XML tags using regex
    text = re.sub(r"<[^>]+>", " ", content)
    words = text.split()
    # Filter out words consisting only of digits and punctuation
    filtered = [w for w in words if not is_digit_and_punct(w)]
    return " ".join(filtered)

# RDF cleaning reuses XML cleaning logic
clean_rdf_file = clean_xml_file

def extract_keyphrases(text):
    """
    Uses the KeyBART model to generate keyphrases from input text.
    Handles input truncation and uses beam search with length constraints
    and repetition penalty for generation.
    """
    global global_tokenizer, global_model, global_device
    if global_tokenizer is None or global_model is None:
        logging.error("Model or tokenizer not initialized in worker process.")
        return [] # Return empty list if model is not ready

    try:
        # Tokenize input text, truncating to 512 tokens
        inputs = global_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(global_device) for k, v in inputs.items()}

        # Generate keyphrases using beam search
        outputs = global_model.generate(
            **inputs,
            num_beams=10,
            max_length=MAX_GENERATE_LENGTH,
            min_length=MIN_GENERATE_LENGTH,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            early_stopping=True
        )
        # Decode the generated tokens back to text
        gen_text = global_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logging.error(f"Error during keyphrase generation: {e}")
        return []

    # Split the generated text into keyphrases based on common separators
    if gen_text:
        phrases = re.split(r"[;,\uFF0C\u3001]+", gen_text) # Handles ;, , ，, 、
        keyphrases = [p.strip() for p in phrases if p.strip()]
    else:
        keyphrases = []

    if len(keyphrases) < MIN_KEYPHRASE_COUNT:
        logging.warning(f"Generated fewer keyphrases than target ({len(keyphrases)} vs {MIN_KEYPHRASE_COUNT}).")

    # Optional: Clean up CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return keyphrases

# ---------------- Result Saving Function ------------------
def save_result_to_db(table_name, result):
    """
    Saves a single processing result (file_id and keyphrases) to the specified database table.
    Uses REPLACE INTO for idempotency.
    """
    if result is None or "file_id" not in result or "keyphrases" not in result:
        logging.warning(f"Attempted to save invalid result to table {table_name}.")
        return
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Ensure table exists (basic check, production might need a more robust migration)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
          file_id VARCHAR(255) PRIMARY KEY,
          keyphrase TEXT
        )
        """
        cursor.execute(create_table_sql)

        file_id = str(result["file_id"])
        keyphrase_json = json.dumps(result["keyphrases"], ensure_ascii=False)
        sql = f"REPLACE INTO {table_name} (file_id, keyphrase) VALUES (%s, %s)"
        cursor.execute(sql, (file_id, keyphrase_json))
        conn.commit()
        # logging.info(f"Saved result for file_id {file_id} to table {table_name}.")

    except mysql.connector.Error as e:
        logging.error(f"Database error while saving result for file_id {result.get('file_id', 'N/A')} to table {table_name}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while saving result for file_id {result.get('file_id', 'N/A')} to table {table_name}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ---------------- Task Specific Processing Functions ------------------

def load_file_metadata():
    """
    Loads file metadata (file_id, data_filename, detect_format) from the database.
    Returns a dictionary keyed by lowercase data_filename.
    """
    conn = None
    cursor = None
    metadata_dict = {}
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        query = f"SELECT file_id, data_filename, detect_format FROM {DB_FILE_META_TABLE}"
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            if "data_filename" in row:
                key = str(row["data_filename"]).lower()
                metadata_dict[key] = row
        logging.info(f"Loaded {len(metadata_dict)} metadata records from {DB_FILE_META_TABLE}.")
    except mysql.connector.Error as e:
        logging.error(f"Database error while loading file metadata from {DB_FILE_META_TABLE}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while loading file metadata from {DB_FILE_META_TABLE}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return metadata_dict

def process_local_file(file_path, db_meta):
    """
    Processes a single local file: determines cleaning method based on
    database metadata, cleans the content, extracts keyphrases, and returns results.
    """
    detect_format = str(db_meta.get("detect_format", "")).lower()
    file_id = db_meta.get("file_id", "N/A")
    file_name = os.path.basename(file_path)

    if detect_format not in GROUP_A and detect_format not in GROUP_B:
        logging.warning(f"Skipping file {file_name} ({file_id}): Unknown detect_format '{detect_format}'.")
        return None

    cleaned_text = ""
    try:
        # Check file size and decide on truncation
        file_size = os.path.getsize(file_path)
        truncated = file_size > FILE_SIZE_THRESHOLD
        if truncated:
            logging.warning(f"File {file_name} ({file_id}) is large ({file_size} bytes), truncating content.")

        # Apply cleaning based on detected format
        if detect_format == "json":
            cleaned_text = clean_json_file(file_path, truncated=truncated)
        elif detect_format == "xml":
            cleaned_text = clean_xml_file(file_path, truncated=truncated)
        elif detect_format in {"xlsx", "xls"}:
            # safe_read_excel handles reading, then clean_table_file handles processing
            cleaned_text = clean_table_file(file_path) # Truncation handled within safe_read_excel if needed (not currently implemented there)
        elif detect_format == "csv":
             cleaned_text = clean_table_file(file_path, truncated=truncated)
        elif detect_format in GROUP_B: # pdf, txt, html, doc, docx treated as plain text
            try:
                with open(file_path, "r", encoding="utf8", errors="ignore") as f:
                    content = f.read(FILE_SIZE_THRESHOLD) if truncated else f.read()
                cleaned_text = clean_text_file(content)
            except FileNotFoundError:
                 logging.error(f"File not found during cleaning {file_name} ({file_id}): {file_path}")
                 return None
            except Exception as e:
                 logging.error(f"Error reading/cleaning text file {file_name} ({file_id}): {e}")
                 return None
        else: # Fallback for formats not explicitly handled but in groups? Should not happen with current group check.
             logging.warning(f"Unhandled format {detect_format} for file {file_name} ({file_id}), treating as plain text.")
             try:
                 with open(file_path, "r", encoding="utf8", errors="ignore") as f:
                     content = f.read(FILE_SIZE_THRESHOLD) if truncated else f.read()
                 cleaned_text = clean_text_file(content)
             except FileNotFoundError:
                  logging.error(f"File not found during fallback cleaning {file_name} ({file_id}): {file_path}")
                  return None
             except Exception as e:
                  logging.error(f"Error reading/cleaning text file (fallback) {file_name} ({file_id}): {e}")
                  return None

    except Exception as e:
        logging.error(f"Error processing file {file_name} ({file_id}): {e}")
        return None

    if not cleaned_text.strip():
        logging.warning(f"Cleaned text is empty for file {file_name} ({file_id}). Skipping keyphrase extraction.")
        return {"file_id": file_id, "keyphrases": []} # Return empty list if no text

    keyphrases = extract_keyphrases(cleaned_text)
    logging.info(f"Extracted {len(keyphrases)} keyphrases for file {file_name} ({file_id}).")

    return {
        "file_id": file_id,
        "file_name": file_name, # Keep file_name for logging/CSV, not saved to DB table
        "keyphrases": keyphrases
    }


def load_acordar_ids():
    """
    Loads distinct file_id values from the rdf_triple_copy1 table.
    """
    conn = None
    cursor = None
    file_ids = []
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        query_file_ids = f"SELECT DISTINCT file_id FROM {DB_TRIPLE_TABLE}"
        cursor.execute(query_file_ids)
        rows = cursor.fetchall()
        file_ids = [row["file_id"] for row in rows if "file_id" in row]
        logging.info(f"Found {len(file_ids)} distinct file_id in {DB_TRIPLE_TABLE}.")
    except mysql.connector.Error as e:
        logging.error(f"Database error while querying distinct file_id from {DB_TRIPLE_TABLE}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while querying distinct file_id from {DB_TRIPLE_TABLE}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return file_ids


def process_database_item(file_id):
    """
    Processes data for a single file_id from database tables
    (rdf_triple_copy1 and rdf_term), reconstructs text from triples,
    extracts keyphrases, and returns results.
    """
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # Query triples for the specific file_id
        query_triples = f"SELECT file_id, subject, predicate, object FROM {DB_TRIPLE_TABLE} WHERE file_id = %s"
        cursor.execute(query_triples, (file_id,))
        triples = cursor.fetchall()

        if not triples:
            logging.warning(f"No triples found for file_id {file_id} in {DB_TRIPLE_TABLE}.")
            cursor.close()
            conn.close()
            return None

        # Query terms for the specific file_id
        query_terms = f"SELECT file_id, term_id, label FROM {DB_TERM_TABLE} WHERE file_id = %s"
        cursor.execute(query_terms, (file_id,))
        terms = cursor.fetchall()

        cursor.close()
        conn.close()

        # Build term_id to label mapping, filtering out invalid labels
        term_mapping = {}
        for row in terms:
            if "term_id" in row and "label" in row:
                term_id = str(row["term_id"]).strip()
                label = row["label"]
                if label is None:
                    continue
                label = str(label).strip()
                if not label or is_digit_and_punct(label):
                    continue
                term_mapping[term_id] = label

        sentences = []
        # Reconstruct sentences from triples using term labels
        for triple in triples:
             if "subject" in triple and "predicate" in triple and "object" in triple:
                sentence_parts = []
                # Get labels for subject, predicate, object from the mapping
                subject_label = term_mapping.get(str(triple["subject"]).strip(), "")
                predicate_label = term_mapping.get(str(triple["predicate"]).strip(), "")
                object_label = term_mapping.get(str(triple["object"]).strip(), "")

                # Append labels if they exist
                if subject_label:
                    sentence_parts.append(subject_label)
                if predicate_label:
                    sentence_parts.append(predicate_label)
                if object_label:
                    sentence_parts.append(object_label)

                # Join parts into a sentence if there are any valid parts
                if sentence_parts:
                    sentence = " ".join(sentence_parts)
                    sentences.append(sentence)

        if not sentences:
            logging.warning(f"Could not construct valid sentences for file_id {file_id}.")
            return {"file_id": file_id, "keyphrases": []} # Return empty if no sentences formed

        # Join sentences into a single text document
        full_text = "\n".join(sentences)

        if not full_text.strip():
            logging.warning(f"Reconstructed text is empty for file_id {file_id}. Skipping keyphrase extraction.")
            return {"file_id": file_id, "keyphrases": []} # Return empty if no text

        # Extract keyphrases from the reconstructed text
        keyphrases = extract_keyphrases(full_text)
        logging.info(f"Extracted {len(keyphrases)} keyphrases for file_id {file_id}.")

        return {
            "file_id": file_id,
            "keyphrases": keyphrases
        }
    except mysql.connector.Error as e:
        logging.error(f"Database error while processing file_id {file_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while processing file_id {file_id}: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ---------------- Main Task Runners ------------------

def run_file_processing_task():
    """
    Runs the local file processing task.
    """
    logging.info("Starting local file processing task...")
    logging.info("Loading file metadata from database...")
    db_metadata = load_file_metadata()
    logging.info(f"Loaded {len(db_metadata)} metadata records.")

    # Prepare list of files to process based on metadata
    to_process = []
    for meta in db_metadata.values():
        detect_format = str(meta.get("detect_format", "")).lower()
        # Determine expected file path based on format grouping
        if detect_format in GROUP_A:
            fp = os.path.join(DATA_FOLDER, meta.get("data_filename", ""))
        elif detect_format in GROUP_B:
            # GROUP_B files are expected to be text files after prior conversion
            fp = os.path.join(DATA_FOLDER, str(meta.get("data_filename", "")) + ".txt")
        else:
            logging.warning(f"File '{meta.get('data_filename', 'N/A')}' with format '{detect_format}' not in allowed groups, skipping.")
            continue

        # Check if the file exists before adding to the list
        if os.path.exists(fp):
            to_process.append((fp, meta))
        else:
            logging.warning(f"File '{fp}' not found, skipping.")

    logging.info(f"Prepared {len(to_process)} files for potential processing.")

    # Load checkpoint for file processing
    processed_set = load_checkpoint(CHECKPOINT_FILE_FILES)
    logging.info(f"Found {len(processed_set)} files marked as processed in checkpoint.")

    # Filter out already processed files
    to_process_filtered = [(fp, meta) for fp, meta in to_process if str(meta.get("file_id")) not in processed_set]
    logging.info(f"Remaining {len(to_process_filtered)} files to process.")

    results = []
    # Determine max workers based on GPU list size
    max_workers = len(GPU_LIST) if GPU_LIST else (os.cpu_count() or 1)
    logging.info(f"Using {max_workers} worker processes.")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(GPU_LIST,)) as executor:
        # Submit processing tasks to the pool
        future_to_info = {
            executor.submit(process_local_file, fp, meta): (fp, meta)
            for fp, meta in to_process_filtered
        }
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(future_to_info), total=len(future_to_info), desc="Processing Local Files"):
            fp, meta = future_to_info[future]
            file_id = meta.get("file_id", "N/A")
            try:
                # Get the result from the completed future
                res = future.result()
                if res is not None:
                    results.append(res)
                    # Add processed file_id to the set and save checkpoint
                    processed_set.add(str(res["file_id"]))
                    save_checkpoint(processed_set, CHECKPOINT_FILE_FILES)
                    # Save result to the database
                    save_result_to_db(DB_RESULT_TABLE_FILES, res)
                    logging.info(f"Finished processing file ID {res['file_id']}, extracted {len(res['keyphrases'])} keyphrases.")
                else:
                    logging.warning(f"Processing file {os.path.basename(fp)} (ID: {file_id}) returned None result.")
            except Exception as e:
                logging.error(f"Exception occurred while processing file {os.path.basename(fp)} (ID: {file_id}): {e}")

    # Final checkpoint save
    save_checkpoint(processed_set, CHECKPOINT_FILE_FILES)
    logging.info("Local file processing task finished.")

    # Optionally save results to CSV
    if results:
        # Filter results to only include items with keyphrases
        results_with_keyphrases = [r for r in results if r and r.get("keyphrases")]
        if results_with_keyphrases:
            df = pd.DataFrame(results_with_keyphrases)
            # Ensure keyphrases list is converted to JSON string for CSV
            df["keyphrase"] = df["keyphrases"].apply(lambda kp: json.dumps(kp, ensure_ascii=False))
            # Select relevant columns for CSV output
            if "file_name" in df.columns:
                df[["file_name", "keyphrase"]].to_csv(RESULT_CSV_FILES, index=False, encoding="utf8")
                logging.info(f"Results saved to CSV file: {RESULT_CSV_FILES}.")
            else:
                 # Handle case where file_name might be missing (though process_local_file adds it)
                 df[["file_id", "keyphrase"]].to_csv(RESULT_CSV_FILES, index=False, encoding="utf8")
                 logging.info(f"Results saved to CSV file: {RESULT_CSV_FILES} (using file_id).")
        else:
             logging.warning("No results with extracted keyphrases to save to CSV.")
    else:
        logging.warning("No results generated for local file processing task, skipping CSV export.")


def run_acordar_processing_task():
    """
    Runs the ACORDAR database processing task.
    """
    logging.info("Starting ACORDAR database processing task...")
    logging.info(f"Loading distinct file_id from {DB_TRIPLE_TABLE}...")
    file_ids = load_acordar_ids()
    logging.info(f"Found {len(file_ids)} unique file IDs in {DB_TRIPLE_TABLE}.")

    # Load checkpoint for ACORDAR processing
    processed_set = load_checkpoint(CHECKPOINT_FILE_ACORDAR)
    logging.info(f"Found {len(processed_set)} ACORDAR file IDs marked as processed in checkpoint.")

    # Filter out already processed file IDs
    file_ids_to_process = [fid for fid in file_ids if str(fid) not in processed_set]
    logging.info(f"Remaining {len(file_ids_to_process)} ACORDAR file IDs to process.")

    results = []
    # Determine max workers based on GPU list size
    max_workers = len(GPU_LIST) if GPU_LIST else (os.cpu_count() or 1)
    logging.info(f"Using {max_workers} worker processes.")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(GPU_LIST,)) as executor:
        # Submit processing tasks to the pool
        future_to_id = {
            executor.submit(process_database_item, fid): fid
            for fid in file_ids_to_process
        }
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing ACORDAR Data"):
            fid = future_to_id[future]
            try:
                # Get the result from the completed future
                res = future.result()
                if res is not None:
                    results.append(res)
                    # Add processed file_id to the set and save checkpoint
                    processed_set.add(str(res["file_id"]))
                    save_checkpoint(processed_set, CHECKPOINT_FILE_ACORDAR)
                    # Save result to the database
                    save_result_to_db(DB_RESULT_TABLE_ACORDAR, res)
                    logging.info(f"Finished processing ACORDAR file ID {res['file_id']}, extracted {len(res['keyphrases'])} keyphrases.")
                else:
                     # Process_database_item might return None for various reasons (e.g., no triples)
                    logging.warning(f"Processing ACORDAR file ID {fid} returned None result.")
                    # Optionally add to processed_set even if result is None, to avoid reattempting failed/empty IDs
                    # processed_set.add(str(fid))
                    # save_checkpoint(processed_set, CHECKPOINT_FILE_ACORDAR) # Only save checkpoint if added to set
            except Exception as e:
                logging.error(f"Exception occurred while processing ACORDAR file ID {fid}: {e}")

    # Final checkpoint save
    save_checkpoint(processed_set, CHECKPOINT_FILE_ACORDAR)
    logging.info("ACORDAR database processing task finished.")

    # Optionally save results to CSV
    if results:
         # Filter results to only include items with keyphrases
        results_with_keyphrases = [r for r in results if r and r.get("keyphrases")]
        if results_with_keyphrases:
            df = pd.DataFrame(results_with_keyphrases)
            # Ensure keyphrases list is converted to JSON string for CSV
            df["keyphrase"] = df["keyphrases"].apply(lambda kp: json.dumps(kp, ensure_ascii=False))
            # Select relevant columns for CSV output (only file_id and keyphrase for ACORDAR)
            df[["file_id", "keyphrase"]].to_csv(RESULT_CSV_ACORDAR, index=False, encoding="utf8")
            logging.info(f"Results saved to CSV file: {RESULT_CSV_ACORDAR}.")
        else:
             logging.warning("No results with extracted keyphrases to save to CSV for ACORDAR task.")
    else:
        logging.warning("No results generated for ACORDAR task, skipping CSV export.")


# ---------------- Main Entry Point ------------------

if __name__ == '__main__':
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Script started.")

    # Run the local file processing task
    run_file_processing_task()

    # Add a separator or delay if desired between tasks
    logging.info("-" * 50)

    # Run the ACORDAR database processing task
    run_acordar_processing_task()

    logging.info("Script finished.")