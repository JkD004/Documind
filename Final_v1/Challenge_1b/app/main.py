import os
import json
import datetime
from pdf_processor import PDFProcessor
from nlp_analyzer import NLPAnalyzer
from ranker import Ranker

# --- IMPORTANT PATH ADJUSTMENT FOR LOCAL RUNNING ---
# Get the directory where the main.py script is located (i.e., 'app' directory)
CURRENT_FILE_PATH = os.path.abspath(__file__)
APP_DIR = os.path.dirname(CURRENT_FILE_PATH)

# Navigate up one level from 'app' to get to the project root (Challenge_1b/)
# This is where your 'Collection X' folders reside.
PROJECT_ROOT_DIR = os.path.dirname(APP_DIR)

# Set INPUT_DIR to the project root so it can discover 'Collection X' folders
INPUT_DIR = PROJECT_ROOT_DIR
# ---------------------------------------------------

def process_collection(collection_path):
    """
    Processes a single document collection, extracting information based on persona and job-to-be-done.
    """
    
    # Construct path to the input JSON file within the collection folder
    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON not found for collection at {collection_path}")
        return

    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    persona = input_data["persona"]["role"]
    job_to_be_done = input_data["job_to_be_done"]["task"]
    input_documents_meta = input_data["documents"]
    
    processed_pdfs = []
    
    # Construct path to the 'PDFs' subfolder within the current collection
    pdf_files_path = os.path.join(collection_path, "PDFs")
    if not os.path.exists(pdf_files_path):
        print(f"Error: 'PDFs' subfolder not found in {collection_path}")
        return

    # Iterate through the documents listed in challenge1b_input.json
    for doc_meta in input_documents_meta:
        filename = doc_meta["filename"]
        
        # Construct the full path to each PDF file
        pdf_path = os.path.join(pdf_files_path, filename)
        
        if os.path.exists(pdf_path):
            print(f"Processing PDF: {filename}")
            # Call your PDFProcessor to extract structured data and full text
            pdf_data = PDFProcessor.extract_structured_data(pdf_path) 
            processed_pdfs.append({
                "filename": filename,
                "title": pdf_data.get("title", ""),
                "outline": pdf_data.get("outline", []),
                "full_text": pdf_data.get("full_text", "") 
            })
        else:
            print(f"Warning: PDF file not found: {pdf_path}. Skipping.")

    # Initialize NLP analyzer with persona and job-to-be-done
    nlp_analyzer = NLPAnalyzer(persona, job_to_be_done)
    
    # Analyze documents to extract relevant sections based on NLP
    extracted_sections_raw = nlp_analyzer.analyze_documents(processed_pdfs)

    # Rank the extracted sections by importance
    ranker = Ranker()
    ranked_sections = ranker.rank(extracted_sections_raw)
    
    # Perform sub-section analysis on the ranked sections
    subsection_analysis_results = nlp_analyzer.perform_subsection_analysis(ranked_sections, processed_pdfs)

    # Prepare the final output JSON structure as per challenge requirements
    output_data = {
        "metadata": {
            "input_documents": [d["filename"] for d in input_documents_meta],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": ranked_sections,  # Use ranked_sections here
        "subsection_analysis": subsection_analysis_results  # Use subsection_analysis_results here
    }

    # Save the output JSON directly within the collection folder
    output_json_path = os.path.join(collection_path, "challenge1b_output.json")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Output saved to {output_json_path}")


if __name__ == "__main__":
    print(f"Starting Challenge 1B processing. Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scanning for collections in: {INPUT_DIR}")
    
    # Iterate through items in the PROJECT_ROOT_DIR (which is now INPUT_DIR)
    for item_name in os.listdir(INPUT_DIR):
        potential_collection_path = os.path.join(INPUT_DIR, item_name)
        
        # Check if it's a directory and starts with "Collection" to identify a challenge collection
        if os.path.isdir(potential_collection_path) and item_name.startswith("Collection"):
            print(f"\n--- Processing Collection: {item_name} ---")
            process_collection(potential_collection_path)
        # else:
            # Optionally, you could print which items are being skipped if they don't match the pattern.
            # print(f"Skipping: {item_name} (not a recognized collection folder)")

    print("\nAll accessible collections processed. Challenge 1B execution complete.")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")