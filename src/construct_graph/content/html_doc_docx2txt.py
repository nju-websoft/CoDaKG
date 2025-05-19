import logging
import pymysql
import os
from bs4 import BeautifulSoup
from docx import Document

# This import requires pypiwin32 and Microsoft Word installed on a Windows system
# It is necessary for converting .doc files using COM automation.
import win32com.client


# Log file configuration
LOG_FILE = 'error_log.log'
LOG_LEVEL = logging.ERROR # Set minimum logging level

# Input and output directories for file conversions
INPUT_DOC_FOLDER = './doc'
OUTPUT_TXT_FOLDER_DOC = './doc2txt' # Output folder for text extracted from .doc files
INPUT_HTML_FOLDER = './html'
OUTPUT_TXT_FOLDER_HTML = './html2txt'
INPUT_DOCX_FOLDER = './docx'
OUTPUT_TXT_FOLDER_DOCX = './docx2txt'

# Constants for file extensions
TEXT_EXTENSION = '.txt'

# Encoding for reading/writing text files
TEXT_ENCODING = 'utf-8'

# ---------------- Logging Setup ------------------
# Configure logging to capture errors to a file and the console
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE), # Log messages to a file
        logging.StreamHandler() # Log messages to the console
    ]
)

# ---------------- File Conversion Functions ------------------

def convert_html_to_txt(file_path: str, target_path: str) -> bool:
    """
    Converts an HTML file to plain text using BeautifulSoup.
    Saves the output text to the target path.
    Returns True on success, False on error.
    """
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Read HTML content
        with open(file_path, "r", encoding=TEXT_ENCODING) as f:
            html_content = f.read()
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, "lxml")
        text_content = soup.get_text(separator="\n", strip=True)
        # Write extracted text to file
        with open(target_path, "w", encoding=TEXT_ENCODING) as f:
            f.write(text_content)
        # logging.info(f"Successfully converted {file_path} to {target_path}") # Optional success log
        return True
    except FileNotFoundError:
        logging.error(f"HTML file not found: {file_path}")
        return False
    except Exception as e:
        logging.error(f'Error converting HTML file {file_path}: {e}')
        return False


def convert_docx_to_txt(file_path: str, target_path: str) -> bool:
    """
    Converts a DOCX file to plain text using python-docx.
    Saves the output text to the target path.
    Returns True on success, False on error.
    """
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Open DOCX document
        doc = Document(file_path)
        text = ""
        # Iterate through paragraphs and extract text
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        # Write extracted text to file
        with open(target_path, "w", encoding=TEXT_ENCODING) as f:
            f.write(text)
        # logging.info(f"Successfully converted {file_path} to {target_path}") # Optional success log
        return True
    except FileNotFoundError:
        logging.error(f"DOCX file not found: {file_path}")
        return False
    except Exception as e:
        logging.error(f'Error converting DOCX file {file_path}: {e}')
        return False


def extract_doc_text():
    """
    Extracts text content from .doc files in a specified input folder
    using win32com.client (requires Microsoft Word on Windows)
    and saves the extracted text to an output folder.
    """
    logging.info(f"Starting text extraction from .doc files in {INPUT_DOC_FOLDER}")
    
    # Ensure output directory exists
    try:
        os.makedirs(OUTPUT_TXT_FOLDER_DOC, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {OUTPUT_TXT_FOLDER_DOC}: {e}")
        return # Exit if cannot create output directory

    word = None # Initialize word object to None
    try:
        # Initialize Microsoft Word COM object
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # Keep Word application hidden

        # Iterate through files in the input directory
        for filename in os.listdir(INPUT_DOC_FOLDER):
            # Construct the full file path
            file_path = os.path.abspath(os.path.join(INPUT_DOC_FOLDER, filename))
            
            # Check if it's a file and ends with .doc (case-insensitive)
            if os.path.isfile(file_path) and filename.lower().endswith('.doc'):
                print(f"Processing: {filename}...", end='\t') # Print file name being processed
                try:
                    # Open the .doc document
                    doc = word.Documents.Open(file_path)
                    # Extract all text content
                    text = doc.Content.Text
                    # Close the document without saving changes
                    doc.Close(SaveChanges=win32com.client.constants.wdDoNotSaveChanges)
                    print('Done') # Indicate successful processing

                    # Construct the target path for the output text file
                    base_filename = os.path.splitext(filename)[0] # Get filename without extension
                    target_path = os.path.join(OUTPUT_TXT_FOLDER_DOC, base_filename + TEXT_EXTENSION)

                    # Save the extracted text to a .txt file
                    with open(target_path, 'w', encoding=TEXT_ENCODING) as f:
                        f.write(text)
                    # logging.info(f"Extracted text from {filename} to {target_path}") # Optional success log

                except Exception as e:
                    print(f"Error: {e}") # Print error message for the file
                    logging.error(f"Error processing .doc file {file_path}: {e}")
            else:
                logging.debug(f"Skipping non-.doc file or directory: {filename}")


    except Exception as e:
        logging.error(f"An error occurred with Word automation: {e}")
        print(f"An error occurred: {e}")
    finally:
        # Ensure Word application is closed even if errors occur
        if word:
            try:
                word.Quit()
            except Exception as e:
                logging.error(f"Error quitting Word application: {e}")


# ---------------- Main Execution Block ------------------
if __name__ == '__main__':
    # This is the main entry point of the script.
    # It primarily calls the function to extract text from .doc files.
    
    logging.info("Script started.")

    # Run the process to extract text from .doc files
    # Note: This requires Microsoft Word installed and running on Windows.
    extract_doc_text()

    for filename in os.listdir(INPUT_DOCX_FOLDER):
        # Construct the full file path
        file_path = os.path.abspath(os.path.join(INPUT_DOCX_FOLDER, filename))
        output_file_path = os.path.abspath(os.path.join(OUTPUT_TXT_FOLDER_DOCX, filename)) + '.txt'
        convert_docx_to_txt(file_path, output_file_path)

    for filename in os.listdir(INPUT_HTML_FOLDER):
        # Construct the full file path
        file_path = os.path.abspath(os.path.join(INPUT_HTML_FOLDER, filename))
        output_file_path = os.path.abspath(os.path.join(OUTPUT_TXT_FOLDER_HTML, filename)) + '.txt'
        convert_html_to_txt(file_path, output_file_path)

    logging.info("Script finished.")