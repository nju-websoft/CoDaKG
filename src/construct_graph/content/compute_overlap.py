import json
import pickle
import os
import logging
import csv
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import sys
from typing import Iterable, Tuple, Any, Optional, Union

# --- Constants ---
DEFAULT_NUM_PERM = 128
DEFAULT_LSH_THRESHOLD = 0.5
DEFAULT_NUM_PROCESSES = max(1, os.cpu_count() // 2 if os.cpu_count() else 1) # Ensure cpu_count returns a value
DEFAULT_SAVE_INTERVAL = DEFAULT_NUM_PROCESSES * 20
LOG_FILE_PATH = 'logs/compute_overlap.log'
CSV_FIELD_SIZE_LIMIT = sys.maxsize

# --- CSV field size limit ---
csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)

# --- Logging setup ---
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

def save_minhash_state(signatures: dict, filename: str):
    """Saves the computed MinHash signatures dictionary to a file (Pickle)."""
    try:
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(signatures, f)
        logging.info(f"Saved {len(signatures)} MinHash signatures to {filename}")
    except Exception as e:
        logging.error(f"Error saving MinHash state to {filename}: {e}")

def load_minhash_state(filename: str) -> dict:
    """Loads previously computed MinHash signatures from a Pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                signatures = pickle.load(f)
            if isinstance(signatures, dict):
                logging.info(f"Loaded {len(signatures)} MinHash signatures from {filename}.")
                return signatures
            else:
                logging.error(f"File {filename} contains invalid data (not a dictionary). Starting new calculation.")
                return {}
        except (pickle.UnpicklingError, EOFError, TypeError, Exception) as e:
            logging.error(f"Error loading MinHash state file {filename}: {e}. Starting new calculation.")
            return {}
    else:
        logging.info(f"State file {filename} not found. Starting new calculation.")
        return {}

def _compute_minhash_worker(args: Tuple[Any, Union[list, set], int]) -> Tuple[Any, Optional[MinHash]]:
    """
    Worker function for parallel MinHash computation.
    Input: args - tuple (item_id, item_data, num_perm)
           item_data should be an iterable (set, list) containing items to hash.
    Output: tuple (item_id, minhash_object or None)
    """
    item_id, item_data, num_perm = args
    try:
        m = MinHash(num_perm=num_perm)
        if not hasattr(item_data, '__iter__') or isinstance(item_data, str):
             logging.warning(f"WORKER: Data for item ID {item_id} is not a valid iterable collection (list/set). Skipping.")
             return item_id, None

        if not item_data: # item_data can be an empty set or list
            logging.debug(f"WORKER: Data for item ID {item_id} is empty. Creating empty MinHash.")
        else:
            for item in item_data:
                try:
                    m.update(str(item).encode('utf-8'))
                except Exception as inner_e:
                    logging.warning(f"WORKER: Failed to convert or hash internal element '{item}' for item ID {item_id}: {inner_e}")
                    continue

        return item_id, m

    except Exception as e:
        logging.error(f"WORKER: Error computing MinHash for item ID {item_id}: {e}", exc_info=False)
        return item_id, None

def compute_minhash_signatures(
    data_iterable: Iterable[Tuple[Any, Union[str, list, set]]],
    save_path: str,
    num_perm: int = DEFAULT_NUM_PERM,
    num_processes: int = DEFAULT_NUM_PROCESSES,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    state_file_path: Optional[str] = None
    ) -> Optional[dict]:
    """
    Computes MinHash signatures for a series of data items, supporting streaming via iterators, parallelism, and checkpointing.

    Args:
        data_iterable (Iterable): An iterable that yields tuples of (item_id, item_data_raw).
                                  item_id is a unique identifier.
                                  item_data_raw can be:
                                    - list or set (containing hashable/string-convertible elements)
                                    - str (will be parsed as JSON list/set)
                                  Example: generator yielding ('id1', {'a', 'b'}), ('id2', '["c", "d", "a"]'), ...
        save_path (str): File path to save the final computed MinHash signatures (K-V pairs, .pkl recommended).
        num_perm (int): Number of permutation functions for MinHash.
        num_processes (int): Number of worker processes for parallel computation.
        save_interval (int): Save intermediate progress after every N new items processed.
        state_file_path (str, optional): Path to load and save intermediate progress.
                                         If None, defaults to save_path.
                                         Recommended to set this to differentiate final output from intermediate state.

    Returns:
        dict: A dictionary of {item_id: minhash_object}. Returns None or an empty dict if processing fails or yields no valid signatures.
    """
    if not hasattr(data_iterable, '__iter__'):
        logging.error("Input 'data_iterable' must be an iterable object (e.g., list, generator, iterator).")
        return None

    if state_file_path is None:
        state_file_path = save_path
        logging.warning(f"state_file_path not specified, using save_path ({save_path}) to save intermediate state.")

    logging.info(f"Starting MinHash computation (streaming): NUM_PERM={num_perm}, using {num_processes} processes.")
    logging.info(f"State will be loaded/saved from: {state_file_path}")
    logging.info(f"Final results will be saved to: {save_path}")

    minhash_signatures = load_minhash_state(state_file_path)
    initial_loaded_count = len(minhash_signatures)
    logging.info(f"Loaded {initial_loaded_count} existing signatures.")

    total_items_seen = 0
    items_skipped_existing = 0
    items_skipped_parsing_error = 0
    items_skipped_type_error = 0
    computed_count = 0

    batch_args = []
    batch_size = save_interval if save_interval > 0 else DEFAULT_SAVE_INTERVAL

    logging.info("Starting to stream input data and process in batches...")

    try:
        data_iterator_with_progress = tqdm(data_iterable, desc="Reading input data stream")
        for item_id, item_data_raw in data_iterator_with_progress:
            total_items_seen += 1
            if item_id in minhash_signatures:
                items_skipped_existing += 1
                continue

            processed_data = None
            if isinstance(item_data_raw, (list, set)):
                processed_data = item_data_raw
            elif isinstance(item_data_raw, str):
                try:
                    loaded_json = json.loads(item_data_raw)
                    if isinstance(loaded_json, (list, set)):
                        processed_data = loaded_json
                    else:
                        logging.warning(f"Item ID {item_id}: JSON parsing result is not list/set (type: {type(loaded_json)}). Skipping this item.")
                        items_skipped_type_error += 1
                        continue
                except json.JSONDecodeError as e:
                    logging.error(f"Item ID {item_id}: Error parsing JSON data: {e}. Skipping this item. Raw data: '{str(item_data_raw)[:100]}...'")
                    items_skipped_parsing_error += 1
                    continue
                except Exception as e:
                    logging.error(f"Item ID {item_id}: Unexpected error processing raw data: {e}. Skipping this item.", exc_info=False)
                    items_skipped_parsing_error += 1
                    continue
            else:
                logging.warning(f"Item ID {item_id}: Encountered invalid data type ({type(item_data_raw)}). Expected list, set, or JSON string. Skipping this item.")
                items_skipped_type_error += 1
                continue

            if processed_data is not None:
                batch_args.append((item_id, processed_data, num_perm))

            if len(batch_args) >= batch_size:
                try:
                    results_iterator_batch = process_map(
                        _compute_minhash_worker, batch_args,
                        max_workers=num_processes,
                        chunksize=max(1, len(batch_args) // (num_processes * 4)) if num_processes > 0 else 1,
                        desc="Computing MinHash (batch)"
                    )
                    for res_item_id, m_hash in results_iterator_batch:
                        if res_item_id is not None and m_hash is not None:
                            minhash_signatures[res_item_id] = m_hash
                            computed_count += 1
                        elif res_item_id is not None:
                            logging.warning(f"Received null MinHash result for item ID: {res_item_id}, possibly due to computation error or empty data.")
                    logging.info(f"Processed {computed_count} new items (total {initial_loaded_count + computed_count}), saving progress to {state_file_path}...")
                    save_minhash_state(minhash_signatures, state_file_path)
                except Exception as e:
                    logging.error(f"Error during batch parallel MinHash computation: {e}", exc_info=True)
                batch_args = []

        if batch_args: # Process any remaining items
            try:
                results_iterator_final = process_map(
                    _compute_minhash_worker, batch_args,
                    max_workers=num_processes,
                    chunksize=max(1, len(batch_args) // (num_processes * 4)) if num_processes > 0 else 1,
                    desc="Computing MinHash (final batch)"
                )
                for res_item_id, m_hash in results_iterator_final:
                    if res_item_id is not None and m_hash is not None:
                        minhash_signatures[res_item_id] = m_hash
                        computed_count += 1
                    elif res_item_id is not None:
                        logging.warning(f"Received null MinHash result for item ID: {res_item_id}, possibly due to computation error or empty data.")
                logging.info(f"Processed {computed_count} new items (total {initial_loaded_count + computed_count}), saving progress to {state_file_path}...")
                save_minhash_state(minhash_signatures, state_file_path)
            except Exception as e:
                logging.error(f"Error during final batch parallel MinHash computation: {e}", exc_info=True)
            batch_args = []

    except Exception as e:
        logging.error(f"Error iterating input data stream: {e}", exc_info=True)
        logging.warning("Attempting to continue with already collected items...")

    logging.info(f"Input data iteration complete. Total items seen: {total_items_seen}.")
    logging.info(f"Skipped {items_skipped_existing} existing items, {items_skipped_parsing_error} parsing error items, {items_skipped_type_error} type error items.")
    logging.info(f"Computed {computed_count} new signatures in this run.")

    if state_file_path != save_path or computed_count > 0:
        logging.info(f"Saving final results ({len(minhash_signatures)} signatures) to {save_path}...")
        save_minhash_state(minhash_signatures, save_path)

    final_signatures = {k: v for k, v in minhash_signatures.items() if v is not None}
    logging.info(f"MinHash computation finished. Returning {len(final_signatures)} valid signatures.")
    return final_signatures


def compute_lsh_similarity(
    minhash_signatures: dict,
    save_path: str,
    num_perm: int = DEFAULT_NUM_PERM,
    lsh_threshold: float = DEFAULT_LSH_THRESHOLD
    ) -> Optional[list]:
    """
    Computes Jaccard similarity between items meeting the threshold using MinHash signatures and LSH.

    Args:
        minhash_signatures (dict): Dictionary of {item_id: minhash_object}.
                                   Typically the return value of compute_minhash_signatures or loaded from a file.
        save_path (str): CSV file path to save similarity results.
                         Columns will be: id1, id2, j_sim.
        num_perm (int): Number of permutation functions used when generating MinHash signatures (must match).
        lsh_threshold (float): Jaccard similarity threshold for LSH (0.0 to 1.0).

    Returns:
        list: A list of dictionaries containing similar pair information: [{'id1': ..., 'id2': ..., 'j_sim': ...}, ...]
              Returns None or an empty list if input is invalid or processing fails.
    """
    if not isinstance(minhash_signatures, dict):
        logging.error("Input 'minhash_signatures' must be a dictionary.")
        return None
    
    if not minhash_signatures:
        logging.info("Input MinHash signatures dictionary is empty. No similarity computation needed.")
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id1', 'id2', 'j_sim']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            logging.info(f"Wrote empty similarity file header to {save_path}")
        except IOError as e:
            logging.error(f"Error writing empty results file {save_path}: {e}")
        return []

    try:
        first_val = next(iter(minhash_signatures.values()))
        if not hasattr(first_val, 'jaccard'):
             logging.error("Values in the dictionary do not appear to be valid MinHash objects (missing jaccard method).")
             return None
        minhash_num_perm = getattr(first_val, 'num_perm', None)
        if minhash_num_perm is not None and minhash_num_perm != num_perm:
             logging.warning(f"Provided num_perm ({num_perm}) does not match num_perm in MinHash objects ({minhash_num_perm})! Results might be inaccurate or LSH build might fail.")
    except StopIteration: # Dictionary is empty, handled above
        pass
    except Exception as e:
         logging.error(f"Error checking MinHash objects: {e}")
         return None

    logging.info(f"Starting LSH similarity computation: {len(minhash_signatures)} items, threshold={lsh_threshold}.")
    logging.info(f"Results will be saved to: {save_path}")

    lsh = None
    try:
        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        insert_iterator = tqdm(minhash_signatures.items(), desc="Building LSH index", total=len(minhash_signatures))
        with lsh.insertion_session() as session:
            for item_id, m_hash in insert_iterator:
                if m_hash is not None:
                     minhash_perm_val = getattr(m_hash, 'num_perm', None)
                     if minhash_perm_val is not None and minhash_perm_val != num_perm:
                         logging.error(f"MinHash num_perm ({minhash_perm_val}) for item ID {item_id} does not match LSH ({num_perm})! Skipping insertion.")
                         continue
                     session.insert(item_id, m_hash)
                else:
                    logging.warning(f"Skipping insertion of null MinHash for ID: {item_id}")

    except ValueError as e:
         logging.error(f"Failed to build LSH index: {e}. Ensure LSH num_perm ({num_perm}) matches all MinHash objects' num_perm.", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"Failed to build LSH index: {e}", exc_info=True)
        return None

    if lsh is None:
        logging.error("LSH index object was not successfully initialized or built.")
        return None
    if lsh.is_empty():
         logging.warning("LSH index is empty, no MinHash objects were successfully inserted.")
         try:
             with open(save_path, 'w', newline='', encoding='utf-8') as f:
                 fieldnames = ['id1', 'id2', 'j_sim']
                 writer = csv.DictWriter(f, fieldnames=fieldnames)
                 writer.writeheader()
             logging.info(f"Wrote empty similarity file header to {save_path} (due to empty LSH index)")
         except IOError as e:
             logging.error(f"Error writing empty results file {save_path}: {e}")
         return []

    logging.info("LSH index build complete.")

    results = []
    processed_pairs = set()

    try:
        query_iterator = tqdm(minhash_signatures.items(), desc="Finding similar pairs", total=len(minhash_signatures))
        for item_id, m_hash in query_iterator:
            if m_hash is None:
                continue

            try:
                candidate_ids = lsh.query(m_hash)
            except Exception as e:
                 logging.error(f"Error querying LSH for item ID {item_id}: {e}")
                 continue

            for candidate_id in candidate_ids:
                if item_id == candidate_id:
                    continue

                pair = tuple(sorted((str(item_id), str(candidate_id)))) # Ensure IDs are strings for consistent sorting
                if pair in processed_pairs:
                    continue

                candidate_hash = minhash_signatures.get(candidate_id)
                if candidate_hash:
                    try:
                         if getattr(m_hash, 'num_perm', None) != getattr(candidate_hash, 'num_perm', None):
                             logging.warning(f"num_perm mismatch when computing Jaccard for ({pair[0]}, {pair[1]}). Skipping.")
                             processed_pairs.add(pair)
                             continue
                         j_sim = m_hash.jaccard(candidate_hash)
                         if j_sim >= lsh_threshold:
                             results.append({'id1': pair[0], 'id2': pair[1], 'j_sim': j_sim})
                    except Exception as e:
                         logging.warning(f"Error computing Jaccard for ({pair[0]}, {pair[1]}): {e}")
                    finally:
                        processed_pairs.add(pair)
                else:
                    logging.warning(f"MinHash not found for candidate ID {candidate_id} when computing Jaccard (source: {item_id}).")
                    processed_pairs.add(pair)
    except Exception as e:
         logging.error(f"Error querying LSH or computing Jaccard: {e}", exc_info=True)

    logging.info(f"Querying complete. Found {len(results)} qualifying similar pairs initially.")

    if results:
        logging.info(f"Saving {len(results)} similar pairs to {save_path}...")
        try:
            results.sort(key=lambda x: x['j_sim'], reverse=True)
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id1', 'id2', 'j_sim']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"Similarity results successfully saved to {save_path}")
        except IOError as e:
            logging.error(f"Error writing results to CSV file {save_path}: {e}")
        except Exception as e:
             logging.error(f"Unknown error occurred while saving results: {e}")
    else:
        logging.info(f"No similar pairs found meeting the criteria. Writing an empty file with headers to {save_path}")
        try:
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id1', 'id2', 'j_sim']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            logging.info(f"Wrote empty similarity file header to {save_path}")
        except IOError as e:
            logging.error(f"Error writing empty results file {save_path}: {e}")

    return results