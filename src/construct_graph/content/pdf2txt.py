import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
import pdfplumber # Still needed for finding tables initially, though extraction logic is removed
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import os
import shelve
import multiprocessing
import re

# Constants for configuration and paths
TESSERACT_CMD = r'../bin/tesseract'
PROGRESS_DB_PATH = 'progress_db'
INPUT_FOLDER = 'PDF'
OUTPUT_FOLDER = 'converted_data/converted_text_pdf'
CROPPED_IMAGE_PDF_SUFFIX = '_cropped_image.pdf'
PDF_IMAGE_PNG_SUFFIX = '_PDF_image.png'
TEMP_TEXT_SUFFIX = '_temp_extracted_text.txt'
DEFAULT_GPU_IDS = [0, 1, 2, 3, 4] # Default list of GPU IDs to cycle through

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def text_extraction(element):
    """
    Extracts text content from a text element.
    """
    line_text = element.get_text()
    # The original code also extracted format, but it was not used in the final output logic.
    # Keeping only text extraction to match the utilized logic.
    return line_text

def crop_image(element, pageObj, filename):
    """
    Crops an image element from a PDF page and saves it as a temporary PDF.
    """
    [image_left, image_top, image_right, image_bottom] = [element.x0, element.y0, element.x1, element.y1]
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    cropped_pdf_filename = f'{filename}{CROPPED_IMAGE_PDF_SUFFIX}'
    with open(cropped_pdf_filename, 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)

def convert_to_images(input_file, filename):
    """
    Converts a PDF file (typically a single-page cropped image PDF) to a PNG image.
    """
    images = convert_from_path(input_file)
    if images:
        image = images[0]
        output_file = f"{filename}{PDF_IMAGE_PNG_SUFFIX}"
        image.save(output_file, "PNG")

def image_to_text(image_path):
    """
    Extracts text from an image file using Tesseract.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Removed extract_table and table_converter as they were not used in process_pdf

def process_pdf(filename, gpu_id):
    """
    Processes a single PDF file to extract text content, including text from images.
    Saves extracted content to a text file and tracks progress.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Set GPU for potential use by underlying libraries (e.g., if pdf2image or tesseract were GPU enabled)
    # Note: Standard tesseract and pdf2image often don't use GPUs directly without special builds/configs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    pdf_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{filename}.txt")

    # Check if file has already been processed
    with shelve.open(PROGRESS_DB_PATH) as db:
        processed_files = db.get('processed_files', set())
        if filename in processed_files:
            print(f"File {filename} already processed, skipping.")
            return

    try:
        # Create a PDF file object using PyPDF2 (used for cropping pages)
        pdfFileObj = open(pdf_path, 'rb')
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)

        # Initialize list to store extracted content for the current page
        page_content = []

        # Extract pages using pdfminer for layout analysis
        for pagenum, page in enumerate(extract_pages(pdf_path)):
            # Get the page object from PyPDF2 for cropping
            pageObj = pdfReaded.pages[pagenum]

            # Although table extraction logic is removed, pdfplumber is still opened
            # potentially for other internal reasons in the original design, keeping it.
            pdfplumber.open(pdf_path)

            # Get page elements and sort them by vertical position
            page_elements = [(element.y1, element) for element in page._objs]
            page_elements.sort(key=lambda a: a[0], reverse=True)

            # Process each element on the page
            for i, component in enumerate(page_elements):
                element = component[1]

                # Check if the element is a text container
                if isinstance(element, LTTextContainer):
                    # Extract text from the text element
                    line_text = text_extraction(element)
                    page_content.append(line_text)

                # Check if the element is an image figure
                if isinstance(element, LTFigure):
                    try:
                        # Crop the image element from the PDF page
                        # Pass a copy of pageObj as PyPDF2.PdfReader page objects can be modified
                        crop_image(element, pdfReaded.pages[pagenum], filename)
                        # Convert the cropped image PDF to a PNG image
                        convert_to_images(f'{filename}{CROPPED_IMAGE_PDF_SUFFIX}', filename)
                        # Extract text from the generated image
                        image_text = image_to_text(f'{filename}{PDF_IMAGE_PNG_SUFFIX}')
                        page_content.append(image_text)
                    except Exception as e:
                        print(f"Error processing image in file {filename} on page {pagenum}: {e}")
                        # Continue processing the rest of the page/file

        # Close the PyPDF2 file object
        pdfFileObj.close()

        # Clean up temporary files
        if os.path.exists(f'{filename}{CROPPED_IMAGE_PDF_SUFFIX}'):
            os.remove(f'{filename}{CROPPED_IMAGE_PDF_SUFFIX}')
        if os.path.exists(f'{filename}{PDF_IMAGE_PNG_SUFFIX}'):
            os.remove(f'{filename}{PDF_IMAGE_PNG_SUFFIX}')

        # Write extracted page content to a temporary file
        temp_filename = f'{filename}{TEMP_TEXT_SUFFIX}'
        with open(temp_filename, 'w', encoding='utf-8') as temp_file:
            # The original logic wrote content sequentially, combining text and image text.
            # This loop structure writes the accumulated page_content.
            temp_file.write('\n'.join(page_content))
            temp_file.write('\n\n') # Add extra newlines potentially to align with original temp file logic

        # Process the temporary text file to merge lines separated by empty lines
        with open(temp_filename, 'r', encoding='utf-8') as temp_file:
            lines = temp_file.readlines()

        processed_lines = []
        paragraph = ""

        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                if paragraph:
                    paragraph += " " + stripped_line
                else:
                    paragraph = stripped_line
            else:
                if paragraph:
                    processed_lines.append(paragraph)
                    paragraph = ""

        # Add the last paragraph if it wasn't followed by an empty line
        if paragraph:
            processed_lines.append(paragraph)

        # Remove the temporary file
        os.remove(temp_filename)

        # Write the processed content to the final output file
        with open(output_path, 'w', encoding='utf-8') as final_file:
            for line in processed_lines:
                final_file.write(line + '\n')

        # Record the file as processed
        with shelve.open(PROGRESS_DB_PATH) as db:
            processed_files = db.get('processed_files', set())
            processed_files.add(filename)
            db['processed_files'] = processed_files

        print(f"Text content from file {filename} successfully processed and saved to '{output_path}'.")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        # Record the file as having an error
        with shelve.open(PROGRESS_DB_PATH) as db:
            error_files = db.get('error_files', set())
            error_files.add(filename)
            db['error_files'] = error_files


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get list of PDF files in the input folder
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]

    # Prepare list of GPU IDs to cycle through for multiprocessing
    num_files = len(pdf_files)
    num_gpus = len(DEFAULT_GPU_IDS)
    # Create a list of gpu_ids that cycles through the available IDs for each file
    gpu_ids_for_files = [DEFAULT_GPU_IDS[i % num_gpus] for i in range(num_files)]

    # Use multiprocessing Pool to process PDF files in parallel
    # Assign a GPU ID to each process based on the cycling list
    with multiprocessing.Pool(processes=len(DEFAULT_GPU_IDS)) as pool: # Limit pool size to number of GPUs
        pool.starmap(process_pdf, zip(pdf_files, gpu_ids_for_files))