
---

#  Challenge 1B – PDF Section Extractor with NLP & Docker

This project analyzes PDFs to identify and extract **relevant sections and refined content** based on a **persona's role and goal**, using NLP and OCR techniques. It outputs structured JSON results directly within the input folder.

##  Features

*  Extracts meaningful PDF headings and sections using **font size, indentation, and layout**
*  Ranks sections by **relevance** using **TF-IDF + cosine similarity**
*  Extracts **refined paragraphs** from top-ranked sections
*  Handles **scanned PDFs** via **OCR (Tesseract)**
*  Easy to run in **Docker** on any OS (Windows/Linux/macOS)

---

##  Project Structure

```
Challenge_1b/
├── app/
│   ├── main.py              
│   ├── nlp_analyzer.py     
│   ├── pdf_processor.py     
│   ├── ranker.py            
│   ├── utils.py             
├── Dockerfile               
├── requirements.txt         
├── collection_1/            
│   ├── challenge1b_input.json
│   ├── PDFs/
│   └── challenge1b_output.json 
├── collection_2/            
│   ├── challenge1b_input.json
│   ├── PDFs/
│   └── challenge1b_output.json 
├── collection_3/            
│   ├── challenge1b_input.json
│   ├── PDFs/
│   └── challenge1b_output.json 
```

---

##  Run with Docker

### 1. **Build Docker image**

Run from the project root:

```bash
docker build -t challenge1b .
```

### 2. **Run the container**

Example for folder `collection_1`:

```bash
docker run --rm \
  -v $(pwd)/collection_1:/app/collection_1 \
  challenge1b \
  python -m app.main collection_1
```

 This:

* Mounts `collection_1` into the container
* Reads input from: `collection_1/challenge1b_input.json`
* Outputs result to: `collection_1/challenge1b_output.json`

---

##  Input Format

###  `challenge1b_input.json`:

```json
{
  "persona": {
    "role": "Travel Blogger"
  },
  "job_to_be_done": {
    "task": "Plan a trip to Goa"
  },
  "documents": [
    {
      "filename": "SampleDoc.pdf"
    }
  ]
}
```

* Add your PDFs inside `collection_1/PDFs/`
* Ensure file names match exactly.

---

##  What the Code Does

###  `main.py`

* Takes collection folder as argument (`collection_1`)
* Loads the JSON and PDFs
* Passes everything to `NLPAnalyzer`
* Saves the structured JSON output back into the same folder

###  `NLPAnalyzer`

* Uses `pdf_processor` to extract heading hierarchy
* Maps heading positions to nearby content blocks
* Uses TF-IDF + cosine similarity to rank top 5 sections
* Refines the content from those sections using paragraph-level similarity

###  `pdf_processor`

* Parses PDFs with PyMuPDF (`fitz`)
* Detects headings by:

  * Font size
  * Indentation
  * Bullet formatting
  * OCR fallback for scanned PDFs

###  `ranker`

* Scores each section’s similarity to the user query
* Returns only the top N sections above a similarity threshold

---

##  Sample Output Format

```json
{
  "metadata": {
    "input_documents": ["SampleDoc.pdf"],
    "persona": "Travel Blogger",
    "job_to_be_done": "Plan a trip to Goa",
    "processing_timestamp": "2025-07-28T17:43:21.123Z"
  },
  "extracted_sections": [
    {
      "document": "SampleDoc.pdf",
      "section_title": "Top Places to Visit in Goa",
      "importance_rank": 1,
      "page_number": 3
    },
    ...
  ],
  "subsection_analysis": [
    {
      "document": "SampleDoc.pdf",
      "refined_text": "The best beaches in Goa include Baga, Calangute, and Palolem...",
      "page_number": 3
    },
    ...
  ]
}
```

---

##  Environment Setup (Non-Docker Dev)

If you're not using Docker, install dependencies manually:

```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr
```

Then run:

```bash
python app/main.py collection_1
```

---

## Dependencies

* `PyMuPDF (fitz)` – for PDF parsing
* `pytesseract` – OCR engine
* `OpenCV, PIL` – image pre-processing for OCR
* `scikit-learn` – TF-IDF & cosine similarity
* `tesseract-ocr` – required by OCR (ensure installed in Docker or host)

---
