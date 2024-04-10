import os
import re
import logging
from pdf2image import convert_from_path
from PIL import Image, UnidentifiedImageError
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor

# Set up logging
log_file = "ocr_processing.log"
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_error(message):
    """
    Log an error message with timestamp.
    """
    logging.error(message)

def clean_text(text):
    """
    Clean the OCR extracted text.
    """
    # Modified to preserve non-ASCII characters as Arabic text will have them
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_pdf(pdf_path, langs=["ar"]):
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        log_error(f"Error converting {pdf_path} to images: {e}")
        return None

    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()

    all_text = []

    for page in pages:
        page_image = page.convert('RGB')
        predictions = run_ocr([page_image], [langs], det_model, det_processor, rec_model, rec_processor)

        for pred in predictions:
            # Corrected line
            extracted_text = " ".join([line.text for line in pred.text_lines])
            clean_extracted_text = clean_text(extracted_text)
            all_text.append(clean_extracted_text)

    return ' '.join(all_text)

def save_text(text, filename):
    """
    Save the cleaned text to a file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def update_log(file, log_file):
    """
    Update the log file with the processed file name.
    """
    with open(log_file, 'a') as log:
        log.write(file + '\n')

def load_processed_files(log_file):
    """
    Load the list of already processed files from the log.
    """
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            return set(log.read().splitlines())
    return set()

# Directories and file paths
books_dir = "books"
output_dir = "output_cleaned_texts"
processed_files_log = "processed_files.log"

os.makedirs(output_dir, exist_ok=True)
processed_files = load_processed_files(processed_files_log)
pdf_files = [file for file in os.listdir(books_dir) if file.endswith(".pdf")]

for index, file in enumerate(pdf_files):
    if file in processed_files:
        continue

    pdf_path = os.path.join(books_dir, file)
    print(f"Processing {file}... ({index+1}/{len(pdf_files)})")

    try:
        cleaned_text = process_pdf(pdf_path)
        if cleaned_text:
            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_cleaned.txt")
            save_text(cleaned_text, output_file)
            update_log(file, processed_files_log)
            print(f"Processed and saved {file} ({index+1}/{len(pdf_files)})")
        else:
            print(f"Skipping {file} due to errors.")
            log_error(f"Skipped {file} due to errors in processing.")
    except Exception as e:
        log_error(f"Error processing {file}: {e}")
        print(f"Error processing {file}. Check log for details.")

print("All files processed.")
