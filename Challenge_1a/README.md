# Challenge 1a: PDF Title and Outline Extraction

---

## 🧠 Overview

This repository provides a **Dockerized solution** for extracting structured titles and outlines from PDF documents. It utilizes a hybrid of **vector-based text extraction** (using PyMuPDF) and **OCR fallback** (using Tesseract) to handle both digitally-generated and scanned/image PDFs.

> Designed for **Challenge 1a of the Adobe India Hackathon 2025**.

---

## ✨ Features

- 📌 Smart heading detection using font size, indentation, and layout cues  
- 🧾 OCR fallback for scanned or image-based PDFs  
- 📂 Automatic extraction of document title and hierarchical outlines  
- 🐳 Fully containerized using Docker for reproducibility  
- 📤 JSON output per PDF, conforming to a standardized schema  

---

## 🗂 Directory Structure
---
```
Challenge_1a/
├── sample_dataset/
│ ├── pdfs/ # Input PDF files
│ ├── outputs/ # Output JSON files
│ └── schema/ # Output schema definition
│ └── output_schema.json
├── process_pdfs.py # Main PDF processing script
├── Dockerfile # Docker build configuration
└── README.md # Project documentation
```

---

## 📦 Dependencies

- Python 3.10
- [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- OpenCV
- Pillow
- NumPy

> ✅ All dependencies are automatically installed within the Docker container.

---

## 🚀 Getting Started

### 🔧 1. Build Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor .
```
### 2. **Run the container**

Example for folder `collection_1`:
```bash
docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/sample_dataset/pdfs:ro \
  -v $(pwd)/sample_dataset/outputs:/app/sample_dataset/outputs \
  pdf-outline-extractor
```

-📥 All PDFs in sample_dataset/pdfs/ will be processed.
-📤 Output JSONs will be saved in sample_dataset/outputs/.


###🔍 How It Works
*🧠 Uses vector text spans (font size, boldness, indentation) to identify heading candidates

*🔁 Fallbacks to OCR for scanned or image-based PDFs (Tesseract)

*🎯 Title is detected from the top 1/3 of the first page using large or bold text

*📑 Handles multi-line headings and bullet merges

##📈 Performance & Constraints
*⚙️ Pure CPU-based solution (no GPU required)

*🖥️ Works on AMD64 (x86) systems

*🚀 Optimized to process PDFs < 50 pages in under 10 seconds (system-dependent)

*📡 Does not require internet access

##✅ Validation Checklist
 *Extracts titles and hierarchical outlines

* Supports both vector-based and scanned PDFs

* Fully containerized, no manual intervention required

* Schema-compliant JSON outputs

 *Automatically processes all .pdf files in the input directory

