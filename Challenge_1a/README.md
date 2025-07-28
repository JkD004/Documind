```markdown
# Challenge 1a: PDF Title and Outline Extraction

---

##  Overview

Our repository provides a Dockerized solution for **extracting structured titles and outlines from PDF documents** using a hybrid of vector-based text extraction and OCR (Tesseract). It is built for Challenge 1a of the Adobe India Hackathon 2025.

---

##  Features

- Smart heading detection based on font size, indentation, and layout
-  OCR fallback for image-based or scanned PDFs
-  Automatic extraction of document titles and hierarchical outlines
-  Fully containerized using Docker for reproducibility
-  Outputs per-PDF JSON files conforming to the expected schema

---

##  Directory Structure

```

Challenge\_1a/
├── sample\_dataset/
│   ├── pdfs/            \# Input PDF files
│   ├── outputs/         \# Output JSON files
│   └── schema/          \# Output schema definition (if applicable)
│       └── output\_schema.json
├── process\_pdfs.py      \# Main PDF processing script
├── Dockerfile           \# Docker build configuration
└── README.md            \# Project documentation

````

---

##  Dependencies

- Python 3.10
- [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- OpenCV
- Pillow
- NumPy

> All dependencies are automatically installed via Docker.

---

##  Getting Started

### 1️ Build Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor .
````

### 2️ Run the Container

```bash
docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/sample_dataset/pdfs:ro \
  -v $(pwd)/sample_dataset/outputs:/app/sample_dataset/outputs \
  pdf-outline-extractor
```

 All PDFs in `sample_dataset/pdfs/` will be processed.
 JSON output will be saved to `sample_dataset/outputs/`.

-----

##  Output Format

Each `.json` file (one per PDF) has the following structure:

```json
{
  "title": "Detected Title of the Document",
  "outline": [
    {
      "level": "H1",
      "text": "Main Section Title",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Subsection Title",
      "page": 2
    }
    // ... more items
  ]
}
```

Output conforms to the schema defined in `sample_dataset/schema/output_schema.json`.

-----

## How It Works

  - Uses vector text spans (font size, indentation) from PDFs to detect heading candidates
  - Falls back to OCR if vector data is missing or incomplete
  - Titles are extracted from the top 1/3 of the first page, prioritizing bold or large text
  - Merges consecutive bullet points intelligently to form full heading lines

-----

## Performance & Constraints

  -  Pure CPU solution (no GPU or internet required)
  -  Works on AMD64 (x86) architecture
  -  Optimized for processing PDFs \< 50 pages in under 10 seconds (depends on system)

-----

##  Validation Checklist

  - Extracts titles and hierarchical outlines
  - Works with both vector-based and scanned/image PDFs
  - Fully containerized and self-contained
  - Schema-compliant JSON output per PDF
  - Processes all `.pdf` files in the input directory
  - Requires no manual input or modification

-----

```