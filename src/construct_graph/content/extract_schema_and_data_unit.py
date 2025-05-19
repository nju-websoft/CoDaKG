# process_data.py
import os
import logging
import csv
import json
import xml.etree.ElementTree as ET
import pandas as pd
import pymysql
from typing import Iterable, Tuple, Any, Optional, Union, List, Dict, Set
import re # For text_overlap_run

# Import functions from the first script
from compute_overlap import compute_minhash_signatures, compute_lsh_similarity

# --- Constants ---
# General
BASE_OUTPUT_DIR = './output'
LOG_FILE_PATH_PROCESS = 'logs/process_data.log'
DEFAULT_NUM_PERM_PROCESS = 128 # Should match compute_overlap or be passed
DEFAULT_LSH_THRESHOLD_PROCESS = 0.5 # Should match compute_overlap or be passed
DEFAULT_NUM_PROCESSES_PROCESS = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
DEFAULT_SAVE_INTERVAL_PROCESS = DEFAULT_NUM_PROCESSES_PROCESS * 20

# Database related
MYSQL_CONFIG = {
    "host": "...",
    "port": 3306,
    "user": "...",
    "password": "...",
    "database": "..."
}
DB_DATA_DIR = 'data_search_e_data' # Root directory for data files from DB

# File type specific
SUPPORTED_JSON_XML_FORMATS = ('json', 'xml')
SUPPORTED_TABLE_FORMATS = ('csv', 'xls', 'xlsx')

# Output file names (examples, can be customized per function call)
JSON_XML_CONTENT_MINHASH_FILE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_content_minhashes.pkl')
JSON_XML_CONTENT_STATE_FILE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_content_minhashes.state.pkl')
JSON_XML_CONTENT_SIMILARITY_FILE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_content_similarity.csv')

JSON_XML_PATTERN_MINHASH_FILE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_pattern_minhashes.pkl')
JSON_XML_PATTERN_STATE_FILE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_pattern_minhashes.state.pkl')
JSON_XML_PATTERN_SIMILARITY_FILE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_pattern_similarity.csv')
JSON_XML_PATTERN_TEMP_PICKLE = os.path.join(BASE_OUTPUT_DIR, 'json_xml_pattern_temp_extracted.pkl') # For intermediate EDPs

TABLE_CONTENT_MINHASH_FILE = os.path.join(BASE_OUTPUT_DIR, 'table_content_minhashes.pkl')
TABLE_CONTENT_STATE_FILE = os.path.join(BASE_OUTPUT_DIR, 'table_content_minhashes.state.pkl')
TABLE_CONTENT_SIMILARITY_FILE = os.path.join(BASE_OUTPUT_DIR, 'table_content_similarity.csv')

TEXT_NEW_MINHASH_FILE = os.path.join(BASE_OUTPUT_DIR, "text_new_minhashes.pkl")
TEXT_NEW_STATE_FILE = os.path.join(BASE_OUTPUT_DIR, "text_new_minhashes.state.pkl")
TEXT_NEW_SIMILARITY_FILE = os.path.join(BASE_OUTPUT_DIR, "text_new_similarity_results.csv")

RDF_PATTERN_MINHASH_FILE = os.path.join(BASE_OUTPUT_DIR, 'rdf_pattern_minhashes.pkl')
RDF_PATTERN_STATE_FILE = os.path.join(BASE_OUTPUT_DIR, 'rdf_pattern_temp.state.pkl')
RDF_PATTERN_SIMILARITY_FILE = os.path.join(BASE_OUTPUT_DIR, 'rdf_pattern_similarity_results.csv')


# --- Logging setup ---
os.makedirs(os.path.dirname(LOG_FILE_PATH_PROCESS), exist_ok=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH_PROCESS),
        logging.StreamHandler()
    ]
)

# --- Helper: Database file list loading ---
def load_file_list_from_mysql(supported_formats: Tuple[str, ...]) -> List[Tuple[str, str, str]]:
    """Loads file list from MySQL for given formats (file_id, detect_format, full_path)."""
    results_with_path = []
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        with conn.cursor() as cursor:
            query = """
                SELECT file_id, detect_format, data_filename
                FROM ntcir_datafile
                WHERE detect_format IN %s
            """
            cursor.execute(query, (supported_formats,))
            db_results = cursor.fetchall()
        conn.close()

        for file_id, detect_format, data_filename in db_results:
            if data_filename and detect_format in supported_formats:
                full_path = os.path.join(DB_DATA_DIR, data_filename)
                if not os.path.exists(full_path):
                     logging.warning(f"File not found on disk, skipping: {full_path} (ID: {file_id})")
                     continue
                results_with_path.append((str(file_id), str(detect_format), str(full_path))) # Ensure string IDs
            else:
                logging.warning(f"Skipping DB entry (invalid format or missing filename): ID {file_id}, Format {detect_format}, Filename {data_filename}")
        logging.info(f"Loaded {len(results_with_path)} valid file entries for formats {supported_formats} from MySQL.")
    except pymysql.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Failed to connect to MySQL or query failed: {e}")
    return results_with_path

# --- JSON/XML Content Extraction ---
def flatten_json(data: Any, parent_key: str = '', sep: str ='.') -> List[str]:
    """Recursively flattens JSON object/list into 'path=value' strings."""
    items = []
    if isinstance(data, dict):
        sorted_keys = sorted(data.keys())
        for k in sorted_keys:
            v = data[k]
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}[{i}]" if parent_key else f"[{i}]"
            items.extend(flatten_json(v, new_key, sep=sep))
    else:
        items.append(f"{parent_key}={str(data)}")
    return items

def load_and_extract_json_content(file_path: str) -> Optional[List[str]]:
    """Loads JSON file, extracts content items as a list of strings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            if not first_char:
                logging.info(f"JSON file is empty: {file_path}")
                return []
            f.seek(0)
            data = json.load(f)
        return flatten_json(data)
    except json.JSONDecodeError as e:
        logging.warning(f"Cannot parse JSON file {file_path}: {e}")
        return None
    except FileNotFoundError:
        logging.warning(f"JSON file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Unknown error reading/processing JSON file {file_path}: {e}")
        return None

def traverse_xml_content(element: ET.Element, current_path: str = '', sep: str = '/') -> List[str]:
    """Recursively traverses XML element tree, extracting 'path=value' or 'path[@attribute]=value' strings."""
    items = []
    tag_name = element.tag.split('}')[-1]
    element_path = f"{current_path}{sep}{tag_name}" if current_path else tag_name

    sorted_attrs = sorted(element.attrib.items())
    for name, value in sorted_attrs:
        attr_name = name.split('}')[-1]
        items.append(f"{element_path}[@{attr_name}]={value}")

    text = element.text.strip() if element.text else ""
    if text:
        items.append(f"{element_path}={text}")

    children = sorted(element, key=lambda x: x.tag)
    for child in children:
        items.extend(traverse_xml_content(child, element_path, sep=sep))

    tail = element.tail.strip() if element.tail else ""
    if tail:
        items.append(f"{element_path}/tail={tail}")
    return items

def load_and_extract_xml_content(file_path: str) -> Optional[List[str]]:
    """Loads XML file, extracts content items as a list of strings."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        if root is None:
             logging.info(f"XML file root element is empty: {file_path}")
             return []
        return traverse_xml_content(root)
    except ET.ParseError as e:
        logging.warning(f"Cannot parse XML file {file_path}: {e}")
        return None
    except FileNotFoundError:
        logging.warning(f"XML file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Unknown error reading/processing XML file {file_path}: {e}")
        return None

def generate_json_xml_content_data(file_infos: List[Tuple[str, str, str]]) -> Iterable[Tuple[str, List[str]]]:
    """Generator to yield (file_id, content_items) for JSON/XML files."""
    for file_id, detect_format, file_path in tqdm(file_infos, desc="Extracting JSON/XML content"):
        content_items = None
        if detect_format == 'json':
            content_items = load_and_extract_json_content(file_path)
        elif detect_format == 'xml':
            content_items = load_and_extract_xml_content(file_path)

        if content_items is not None: # Could be an empty list for empty files
            yield file_id, content_items
        else:
            logging.warning(f"Skipping file {file_id} ({file_path}) due to content extraction error.")


# --- JSON/XML Pattern (EDP) Extraction ---
def get_json_paths(data: Any, parent_key: str = '', sep: str = '.') -> Set[str]:
    """Recursively collects all unique JSON paths."""
    paths = set()
    if isinstance(data, dict):
        sorted_keys = sorted(data.keys())
        for k in sorted_keys:
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            paths.add(new_key)
            paths.update(get_json_paths(data[k], new_key, sep=sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}[{i}]" if parent_key else f"[{i}]"
            paths.add(new_key)
            paths.update(get_json_paths(v, new_key, sep=sep))
    return paths

def extract_json_edp(file_path: str) -> Optional[Set[str]]:
    """Loads JSON file and extracts EDP (set of all unique paths)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            if not first_char:
                 logging.info(f"JSON file is empty for EDP: {file_path}")
                 return set()
            f.seek(0)
            data = json.load(f)
        return get_json_paths(data)
    except json.JSONDecodeError as e:
        logging.warning(f"Cannot parse JSON file for EDP {file_path}: {e}")
        return None
    except FileNotFoundError:
         logging.warning(f"JSON file not found for EDP: {file_path}")
         return None
    except Exception as e:
        logging.warning(f"Error extracting JSON EDP from {file_path}: {e}")
        return None

def get_xml_paths(element: ET.Element, current_path: str = '', sep: str = '/') -> Set[str]:
    """Recursively collects all unique XPath-style paths."""
    paths = set()
    tag_name = element.tag.split('}')[-1]
    element_path = f"{current_path}{sep}{tag_name}" if current_path else tag_name
    paths.add(element_path)

    sorted_attrs = sorted(element.attrib.keys())
    for name in sorted_attrs:
        attr_name = name.split('}')[-1]
        paths.add(f"{element_path}[@{attr_name}]")

    children = sorted(element, key=lambda x: x.tag)
    for child in children:
        paths.update(get_xml_paths(child, element_path, sep=sep))
    return paths

def extract_xml_edp(file_path: str) -> Optional[Set[str]]:
    """Loads XML file and extracts EDP (set of all unique structural paths)."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        if root is None:
            logging.info(f"XML file root element is empty for EDP: {file_path}")
            return set()
        return get_xml_paths(root)
    except ET.ParseError as e:
         logging.warning(f"Cannot parse XML file for EDP {file_path}: {e}")
         return None
    except FileNotFoundError:
         logging.warning(f"XML file not found for EDP: {file_path}")
         return None
    except Exception as e:
        logging.warning(f"Error extracting XML EDP from {file_path}: {e}")
        return None

def _extract_edp_worker(args: Tuple[str, str, str]) -> Tuple[str, Optional[Set[str]]]:
    """Worker for parallel EDP extraction."""
    file_id, detect_format, file_path = args
    edp_set = None
    if detect_format == 'json':
        edp_set = extract_json_edp(file_path)
    elif detect_format == 'xml':
        edp_set = extract_xml_edp(file_path)
    else:
        logging.warning(f"WORKER_EDP: Skipping file (unsupported format for EDP): ID {file_id}, Format {detect_format}")
    return file_id, edp_set

def generate_json_xml_pattern_data(file_infos: List[Tuple[str, str, str]],
                                   temp_pickle_file: str,
                                   num_processes: int,
                                   save_interval: int) -> Iterable[Tuple[str, List[str]]]:
    """
    Extracts EDPs in parallel, saves/loads intermediate results, and yields (file_id, sorted_edp_list).
    Uses a temporary pickle file to store all extracted EDPs before yielding.
    """
    if os.path.exists(temp_pickle_file):
        logging.info(f"Loading previously extracted EDPs from {temp_pickle_file}")
        with open(temp_pickle_file, 'rb') as f:
            edp_results_map = pickle.load(f)
    else:
        edp_results_map = {}

    files_to_process_edp = [f_info for f_info in file_infos if f_info[0] not in edp_results_map]

    if files_to_process_edp:
        logging.info(f"Extracting EDPs for {len(files_to_process_edp)} new files...")
        # Note: process_map is not directly used here as the original logic saved to a dict first.
        # For true streaming, _extract_edp_worker would yield directly.
        # This adaptation maintains the intermediate save logic of json_xml_pattern.py
        processed_count = 0
        # To use process_map effectively here, we might need to rethink the temp_pickle saving strategy slightly
        # or accept that all EDPs are computed before MinHashing starts.
        # The original json_xml_pattern.py computed all EDPs, then did MinHash.
        # We will stick to that pattern for now for this specific function.
        
        # Using a simple loop for clarity with intermediate saving, similar to original hash.py MinHash part
        batch_args_edp = []
        batch_size_edp = save_interval 

        for i, f_info in enumerate(tqdm(files_to_process_edp, desc="Preparing EDP extraction batches")):
            batch_args_edp.append(f_info)
            if len(batch_args_edp) >= batch_size_edp or i == len(files_to_process_edp) -1 :
                if not batch_args_edp: continue # Skip if empty
                from tqdm.contrib.concurrent import process_map as edp_process_map # Local import to avoid conflict
                
                try:
                    results_iterator_edp = edp_process_map(_extract_edp_worker, batch_args_edp,
                                                         max_workers=num_processes,
                                                         chunksize=max(1, len(batch_args_edp) // (num_processes * 4)) if num_processes > 0 else 1,
                                                         desc="Extracting EDPs (batch)")
                    for file_id, edp_set in results_iterator_edp:
                        if edp_set is not None:
                            edp_results_map[file_id] = edp_set # Store the set
                            processed_count +=1
                    
                    logging.info(f"Extracted EDPs for {processed_count} new files. Saving intermediate EDPs to {temp_pickle_file}")
                    with open(temp_pickle_file, 'wb') as f_temp:
                        pickle.dump(edp_results_map, f_temp)

                except Exception as e_edp_batch:
                    logging.error(f"Error during batch EDP extraction: {e_edp_batch}", exc_info=True)
                batch_args_edp = []


        logging.info(f"Finished EDP extraction. Total EDPs in map: {len(edp_results_map)}")
        if not os.path.exists(temp_pickle_file) or processed_count > 0 : # Save if new data or file DNE
             logging.info(f"Saving all extracted EDPs to {temp_pickle_file}")
             with open(temp_pickle_file, 'wb') as f_temp:
                pickle.dump(edp_results_map, f_temp)

    else:
        logging.info("All EDPs already loaded from temporary file.")

    # Yield from the map
    for file_id, edp_set in tqdm(edp_results_map.items(), desc="Yielding EDPs"):
        if edp_set is not None: # edp_set is a set of strings
            yield file_id, sorted(list(edp_set)) # MinHash expects a list/set of strings


# --- Table Content Extraction ---
def preprocess_table_row(row: pd.Series) -> str:
    """Converts a DataFrame row to a canonical string for hashing."""
    return "||".join(str(cell).strip() for cell in row.fillna('').values)

def load_table_file_and_extract_rows(file_path: str, detect_format: str) -> Optional[List[str]]:
    """Loads a table file (CSV, XLS, XLSX) and returns list of preprocessed row strings."""
    df = None
    try:
        if not os.path.exists(file_path):
            logging.warning(f"Table file does not exist: {file_path}")
            return None

        if detect_format == 'csv':
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
                except Exception as e_enc:
                    logging.warning(f"Could not read CSV {file_path} with utf-8 or latin1: {e_enc}")
                    return None
            except pd.errors.EmptyDataError:
                 logging.info(f"CSV file is empty: {file_path}")
                 return []
            except Exception as e_csv:
                logging.warning(f"Error reading CSV {file_path}: {e_csv}")
                return None
        elif detect_format in ['xls', 'xlsx']:
            try:
                df = pd.read_excel(file_path, sheet_name=0) # Reads first sheet by default
            except Exception as e_excel:
                logging.warning(f"Error reading Excel file {file_path}: {e_excel}")
                return None
        else:
            logging.error(f"Unsupported table format {detect_format} for file {file_path}.")
            return None

        if df is None:
            logging.warning(f"DataFrame loaded as None from {file_path}.")
            return None
        if df.empty:
            logging.info(f"Table file {file_path} is empty (no rows).")
            return []

        return [preprocess_table_row(row) for _, row in df.iterrows()]

    except Exception as e:
        logging.error(f"General error loading/processing table file {file_path}: {e}")
        return None

def generate_table_content_data(file_infos: List[Tuple[str, str, str]]) -> Iterable[Tuple[str, List[str]]]:
    """Generator to yield (file_id, list_of_row_strings) for table files."""
    for file_id, detect_format, file_path in tqdm(file_infos, desc="Extracting table content"):
        row_strings = load_table_file_and_extract_rows(file_path, detect_format)
        if row_strings is not None: # Can be empty list for empty tables
            yield file_id, row_strings
        else:
            logging.warning(f"Skipping table file {file_id} ({file_path}) due to row extraction error.")

# --- Text File (Sentences) Extraction ---
def generate_text_sentence_data(folder_path: str) -> Iterable[Tuple[str, Set[str]]]:
    """
    Reads all .txt files in a folder, splits content into sentences, and yields (filename, set_of_sentences).
    """
    if not os.path.exists(folder_path):
        logging.error(f"Folder '{folder_path}' does not exist!")
        return
    if not os.path.isdir(folder_path):
        logging.error(f"Path '{folder_path}' is not a directory!")
        return

    for file_name in tqdm(os.listdir(folder_path), desc="Extracting text sentences"):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                sentences = re.split(r'[。！？.!?]+', content)
                # Remove leading/trailing whitespace and empty strings
                processed_sentences = {s.strip() for s in sentences if s.strip()}
                yield file_name, processed_sentences
            except Exception as e:
                logging.error(f"Error reading text file {file_name}: {e}", exc_info=True)
        else:
            logging.debug(f"Skipping non-txt file or directory: {file_name}")


# --- RDF Pattern (from TXT file) Extraction ---
def generate_rdf_pattern_data(rdf_pattern_file_path: str) -> Iterable[Tuple[str, Set[str]]]:
    """
    Reads a tab-separated file (file_id \t comma-separated-edps) and yields (file_id, set_of_edps).
    """
    if not os.path.exists(rdf_pattern_file_path):
        logging.error(f"RDF pattern file '{rdf_pattern_file_path}' not found!")
        return
    
    try:
        with open(rdf_pattern_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting RDF patterns"):
                line_data = line.strip().split('\t')
                if len(line_data) == 2:
                    file_id = line_data[0]
                    edps_str = line_data[1]
                    edps = set(e.strip() for e in edps_str.split(',') if e.strip())
                    yield file_id, edps
                elif line.strip(): # Non-empty line but wrong format
                    logging.warning(f"Skipping malformed line in RDF pattern file: {line.strip()}")
    except Exception as e:
        logging.error(f"Error reading or processing RDF pattern file {rdf_pattern_file_path}: {e}")


# --- Generic Runner Function ---
def run_similarity_pipeline(
    data_iterable_generator_func, # Function that returns an iterable of (id, data_set)
    data_source_arg, # Argument for the generator function (e.g., file path, folder path, db info)
    pipeline_name: str,
    minhash_file: str,
    state_file: str,
    similarity_file: str,
    num_perm: int = DEFAULT_NUM_PERM_PROCESS,
    lsh_threshold: float = DEFAULT_LSH_THRESHOLD_PROCESS,
    num_processes: int = DEFAULT_NUM_PROCESSES_PROCESS,
    save_interval: int = DEFAULT_SAVE_INTERVAL_PROCESS,
    **kwargs # For extra args to data_iterable_generator_func if needed
):
    """
    Runs the full MinHash + LSH pipeline for a given data extraction method.
    """
    logging.info(f"--- Starting {pipeline_name} similarity pipeline ---")
    
    # Ensure output directories exist
    for f_path in [minhash_file, state_file, similarity_file]:
        dir_ = os.path.dirname(f_path)
        if dir_: os.makedirs(dir_, exist_ok=True)

    # 1. Generate data iterable
    logging.info(f"[{pipeline_name}] Generating data for MinHash...")
    # Pass kwargs to the generator function if they are relevant for it
    # Example: generate_json_xml_pattern_data needs temp_pickle_file, num_processes, save_interval
    if data_iterable_generator_func.__name__ == "generate_json_xml_pattern_data":
         data_iterable = data_iterable_generator_func(
            data_source_arg, 
            temp_pickle_file=kwargs.get("temp_pickle_file", JSON_XML_PATTERN_TEMP_PICKLE), # Default if not passed
            num_processes=num_processes, # Use the pipeline's num_processes
            save_interval=save_interval  # Use the pipeline's save_interval
            )
    else:
        data_iterable = data_iterable_generator_func(data_source_arg)


    # 2. Compute MinHash signatures
    logging.info(f"[{pipeline_name}] Computing MinHash signatures...")
    signatures = compute_minhash_signatures(
        data_iterable=data_iterable,
        save_path=minhash_file,
        state_file_path=state_file,
        num_perm=num_perm,
        num_processes=num_processes,
        save_interval=save_interval
    )

    if signatures is None or not signatures: # Check if signatures is None or empty
        logging.error(f"[{pipeline_name}] MinHash signature computation failed or returned no signatures. Halting pipeline.")
        return

    # 3. Compute LSH similarity
    logging.info(f"[{pipeline_name}] Computing LSH similarity...")
    similarity_pairs = compute_lsh_similarity(
        minhash_signatures=signatures,
        save_path=similarity_file,
        num_perm=num_perm,
        lsh_threshold=lsh_threshold
    )

    if similarity_pairs is not None:
        logging.info(f"[{pipeline_name}] Similarity computation complete. Found {len(similarity_pairs)} similar pairs. Results saved to {similarity_file}")
    else:
        logging.error(f"[{pipeline_name}] LSH similarity computation encountered an error.")

    logging.info(f"--- {pipeline_name} similarity pipeline finished ---")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Example: Run JSON/XML Content Overlap
    # This assumes files are listed in a database as in json_xml_content_overlap.py
    json_xml_files_from_db = load_file_list_from_mysql(SUPPORTED_JSON_XML_FORMATS)
    if json_xml_files_from_db:
        run_similarity_pipeline(
            data_iterable_generator_func=generate_json_xml_content_data,
            data_source_arg=json_xml_files_from_db,
            pipeline_name="JSON/XML Content Overlap",
            minhash_file=JSON_XML_CONTENT_MINHASH_FILE,
            state_file=JSON_XML_CONTENT_STATE_FILE,
            similarity_file=JSON_XML_CONTENT_SIMILARITY_FILE
        )
    else:
        logging.warning("No JSON/XML files loaded from DB to process for content overlap.")

    # Example: Run JSON/XML Pattern (EDP) Overlap
    # This also assumes files are listed in a database.
    # The generate_json_xml_pattern_data function handles its own intermediate EDP pickling.
    # json_xml_files_from_db is already loaded above, can reuse if the list is the same.
    if json_xml_files_from_db: # Reuse the loaded list
        run_similarity_pipeline(
            data_iterable_generator_func=generate_json_xml_pattern_data,
            data_source_arg=json_xml_files_from_db,
            pipeline_name="JSON/XML Pattern (EDP) Overlap",
            minhash_file=JSON_XML_PATTERN_MINHASH_FILE,
            state_file=JSON_XML_PATTERN_STATE_FILE,
            similarity_file=JSON_XML_PATTERN_SIMILARITY_FILE,
            # Pass specific kwargs needed by generate_json_xml_pattern_data
            temp_pickle_file=JSON_XML_PATTERN_TEMP_PICKLE,
            # num_processes and save_interval will be taken from the run_similarity_pipeline defaults or its direct args
        )
    else:
        logging.warning("No JSON/XML files loaded from DB to process for pattern overlap.")


    # Example: Run Table Content Overlap
    table_files_from_db = load_file_list_from_mysql(SUPPORTED_TABLE_FORMATS)
    if table_files_from_db:
        run_similarity_pipeline(
            data_iterable_generator_func=generate_table_content_data,
            data_source_arg=table_files_from_db,
            pipeline_name="Table Content Overlap",
            minhash_file=TABLE_CONTENT_MINHASH_FILE,
            state_file=TABLE_CONTENT_STATE_FILE,
            similarity_file=TABLE_CONTENT_SIMILARITY_FILE
        )
    else:
        logging.warning("No table files loaded from DB to process for content overlap.")

    # Example: Run Text New (Sentences) Overlap
    # This expects a folder path containing .txt files
    TEXT_FOLDER_PATH = "text_new" # Replace with the actual path to your "text_new" folder
    if not os.path.isdir(TEXT_FOLDER_PATH):
        logging.warning(f"Text folder '{TEXT_FOLDER_PATH}' not found. Skipping text overlap.")
        # You might want to create a dummy folder for testing if it doesn't exist
        # os.makedirs(TEXT_FOLDER_PATH, exist_ok=True)
        # logging.info(f"Created dummy text folder '{TEXT_FOLDER_PATH}' for testing.")
    else:
         run_similarity_pipeline(
            data_iterable_generator_func=generate_text_sentence_data,
            data_source_arg=TEXT_FOLDER_PATH,
            pipeline_name="Text Sentences Overlap",
            minhash_file=TEXT_NEW_MINHASH_FILE,
            state_file=TEXT_NEW_STATE_FILE,
            similarity_file=TEXT_NEW_SIMILARITY_FILE
        )

    # Example: Run RDF Pattern Overlap (from a TXT file)
    RDF_PATTERN_INPUT_FILE = 'acordar_fileId_global_edpId.txt' # From hash.py example
    if not os.path.exists(RDF_PATTERN_INPUT_FILE):
        logging.warning(f"RDF pattern input file '{RDF_PATTERN_INPUT_FILE}' not found. Skipping RDF pattern overlap.")
        # You might want to create a dummy file for testing
        # with open(RDF_PATTERN_INPUT_FILE, 'w') as f_dummy:
        #     f_dummy.write("rdf_file1\tedp1,edp2,edp3\n")
        #     f_dummy.write("rdf_file2\tedp2,edp3,edp4\n")
        # logging.info(f"Created dummy RDF pattern file '{RDF_PATTERN_INPUT_FILE}' for testing.")
    else:
        run_similarity_pipeline(
            data_iterable_generator_func=generate_rdf_pattern_data,
            data_source_arg=RDF_PATTERN_INPUT_FILE,
            pipeline_name="RDF Pattern Overlap",
            minhash_file=RDF_PATTERN_MINHASH_FILE,
            state_file=RDF_PATTERN_STATE_FILE,
            similarity_file=RDF_PATTERN_SIMILARITY_FILE
        )

    logging.info("All specified processing pipelines have been attempted.")