
import pymysql
import magic
from tqdm import tqdm
import mimetypes
from rdflib import Graph
import json
import os
import csv

# ---------------- Configuration Constants ------------------
# MySQL Database Configuration
MYSQL_CONFIG = {
    "host": "...",
    "port": 3306,
    "user": "...",
    "password": "...",
    "database": "..."
}

# Path to the mimetype mapping JSON file
MIMETYPE_MAPPING_FILE = 'mimetype_mapping.json'

# Directory containing the data files
DATA_DIRECTORY = '../data_search_e_data/'

# Database table to update
DB_TABLE = "ntcir_datafile"

# Set field size limit for CSV operations (although not directly used in this script, good practice if handling large fields)
csv.field_size_limit(sys.maxsize)

# ---------------- Database Connection Helper ------------------
def get_connection():
    """Establishes and returns a database connection using the global config."""
    connection = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        db=MYSQL_CONFIG["database"],
        charset=MYSQL_CONFIG["charset"]
    )
    return connection

# ---------------- File Format Detection Helpers ------------------
def is_rdf(file_path: str) -> bool:
    """
    Attempts to parse a file as RDF using rdflib.
    Returns True if successful and the graph contains triples, False otherwise.
    """
    g = Graph()
    try:
        # Attempt to parse the file, rdflib tries multiple formats
        g.parse(file_path)
        # Check if any triples were loaded into the graph
        if len(g) > 0:
            return True
        else:
            return False
    except Exception:
        # Catch any exception during parsing (e.g., format error, file not found)
        return False

def detect_file_format(filenames: list) -> dict:
    """
    Detects the format of multiple files using 'magic' and 'mimetypes'.
    Uses a predefined mapping for standardization.
    Returns a dictionary mapping file paths to detected formats.
    """
    # Load the mimetype mapping from the JSON file
    try:
        with open(MIMETYPE_MAPPING_FILE, 'r') as f:
            mimetype_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mimetype mapping file not found at {MIMETYPE_MAPPING_FILE}")
        mimetype_mapping = {} # Use empty mapping if file is not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {MIMETYPE_MAPPING_FILE}")
        mimetype_mapping = {} # Use empty mapping if JSON is invalid
    except Exception as e:
        print(f"Error loading mimetype mapping file {MIMETYPE_MAPPING_FILE}: {e}")
        mimetype_mapping = {}

    results_dict = {}
    # Iterate through filenames with a progress bar
    for data_filename in tqdm(filenames, desc='Detecting format'):
        detected_format = 'Unknown'
        magic_mime = 'Unknown'
        mimetypes_mime = 'Unknown'

        # Use python-magic to detect mime type
        try:
            magic_mime = magic.from_file(data_filename, mime=True)
            # Map detected mime type using the loaded mapping
            magic_mime = mimetype_mapping.get(magic_mime, 'Unknown')
        except Exception as e:
             # print(f"Warning: Could not detect format with magic for {data_filename}: {e}")
             pass # Suppress individual file errors unless critical

        # Use mimetypes module to guess mime type from extension
        try:
            mimetypes_guess = mimetypes.guess_type(data_filename)[0]
            if mimetypes_guess:
                 mimetypes_mime = mimetype_mapping.get(mimetypes_guess, 'Unknown')
        except Exception as e:
             # print(f"Warning: Could not detect format with mimetypes for {data_filename}: {e}")
             pass # Suppress individual file errors

        # Determine the final detected format based on results
        if magic_mime != 'Unknown' and magic_mime == mimetypes_mime:
            detected_format = magic_mime
        elif mimetypes_mime != 'Unknown':
            detected_format = mimetypes_mime
        else:
            detected_format = magic_mime # Default to magic if mimetypes fails or is unknown

        results_dict[data_filename] = detected_format

    return results_dict

# ---------------- Main Detection and Update Function ------------------
def detect_format_ntcir(directory: str):
    """
    Detects file formats for files in a directory, checks against existing DB data,
    and updates the 'detect_format' column in the database table.
    """
    # Get a list of all files in the directory with their full paths
    full_file_paths = [os.path.join(directory, x) for x in os.listdir(directory) if os.path.isfile(os.path.join(directory, x))]

    # Detect formats for all files in the directory
    filename_fmt_dict = detect_file_format(full_file_paths)

    conn = None
    cursor = None
    try:
        # Get database connection
        conn = get_connection()
        cursor = conn.cursor()

        # Select files from the database where detect_format is NULL
        # This assumes 'file_id', 'data_format', 'data_filename' columns exist
        sql = f"SELECT file_id, data_format, data_filename FROM {DB_TABLE} WHERE detect_format IS NULL"
        cursor.execute(sql)
        results = cursor.fetchall() # Fetches rows as tuples

        update_data = []
        # Iterate through the database results to prepare update statements
        for file_id, data_format, data_filename in tqdm(results, desc='Processing DB entries'):
            # Construct the full path to the file
            full_data_filename_path = os.path.join(directory, data_filename)
            
            # Get the detected format from the dictionary (use .get() with default for safety)
            detected_format = filename_fmt_dict.get(full_data_filename_path, 'Unknown')

            # Special handling for RDF format
            # If the database format is 'rdf' OR detected format is 'rdf', perform RDF validation
            if data_format == 'rdf' or detected_format == 'rdf':
                if is_rdf(full_data_filename_path):
                    detected_format = 'rdf' # Confirm as rdf if validation passes
                # else: format remains as detected or unknown if RDF validation fails

            # Add the detected format and file_id to the update list
            update_data.append([detected_format, file_id])

        # Prepare the SQL UPDATE statement
        update_sql = f"UPDATE {DB_TABLE} SET detect_format=%s WHERE file_id=%s"

        # Execute batch update
        if update_data:
            try:
                cursor.executemany(update_sql, update_data)
                conn.commit() # Commit the transaction
                print(f"Updated {cursor.rowcount} rows in {DB_TABLE}.")
            except Exception as e:
                print(f"Database update failed: {e}")
                conn.rollback() # Rollback on error
        else:
            print("No database entries found requiring format detection update.")

    except pymysql.MySQLError as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ---------------- Main Execution Block ------------------
if __name__ == '__main__':
    # Run the format detection and database update process
    detect_format_ntcir(directory=DATA_DIRECTORY)