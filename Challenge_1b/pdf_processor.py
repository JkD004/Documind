import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import json
import re
import numpy as np
import cv2
import os

# Regex to detect bullets or list markers at start of lines
BULLET_PATTERN = re.compile(
    r"^(\s*[\u2022\-\\•\◦]|\s\d+[\.\)]|\s*[a-zA-Z]\)|\s*–\s)", re.UNICODE
)

def preprocess_image_for_ocr(image_bytes):
    # Convert bytes to numpy array
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise and threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def ocr_text_from_image(image_bytes):
    processed_img = preprocess_image_for_ocr(image_bytes)

    # Use Tesseract with layout analysis
    custom_config = r'--oem 3 --psm 1'  # Best mode for structured documents
    text = pytesseract.image_to_string(processed_img, config=custom_config)

    return text


def extract_pdf_title(doc):
    first_page = doc.load_page(0)
    blocks = first_page.get_text("dict")["blocks"]
    page_height = first_page.rect.height
    top_limit = page_height / 3  # focus on top third of the page

    spans = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                spans.append({
                    "text": text,
                    "size": round(span.get("size", 0), 2),
                    "flags": span.get("flags", 0),
                    "font": span.get("font", ""),
                    "y": span["bbox"][1],
                    "x_center": (span["bbox"][0] + span["bbox"][2]) / 2
                })

    def is_bold_like(span):
        font = span["font"].lower()
        return "bold" in font or span["flags"] in {1, 4, 5, 33, 37}

    # ========== CASE 1: Use OCR only if NO vector spans ==========
    if not spans:
        pix = first_page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_cv = np.array(img.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(thresh, config=custom_config)
        ocr_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

        def is_valid_title_line(line):
            if len(line) < 6:
                return False
            if len(re.findall(r"[a-zA-Z]", line)) < len(line) * 0.5:
                return False
            if any(re.match(pat, line.lower()) for pat in [
                r"^page\s*\d+$", r"^\d+$", r"^table of contents$", r"^contents?$", r"^index$",
                r"^introduction$", r"^chapter\s+\d+$"
            ]):
                return False
            return True

        clean_lines = [line for line in ocr_lines if is_valid_title_line(line)]

        if len(clean_lines) >= 2 and clean_lines[0][-1] in {":", ",", "-", "–"}:
            return f"{clean_lines[0]} {clean_lines[1]}".strip()
        elif clean_lines:
            return clean_lines[0].strip()

        # Metadata fallback if OCR fails
        meta_title = doc.metadata.get("title", "").strip()
        if meta_title:
            return meta_title

        return "Untitled Document"

    # ========== CASE 2: Use vector span logic ==========
    spans = sorted(spans, key=lambda s: s["y"])
    font_sizes = list({s["size"] for s in spans})
    many_sizes = len(font_sizes) >= 6

    if many_sizes:
        max_size = max(font_sizes)
        title_spans = [
            s for s in spans
            if s["size"] == max_size and s["y"] < top_limit
        ]
        title_text = " ".join([s["text"] for s in title_spans])
        if title_text.strip():
            return title_text.strip()
    else:
        for span in spans:
            if span["y"] < top_limit:
                if is_heading_candidate(span["text"], span["size"], span["flags"], span["y"], None):
                    return span["text"].strip()

        i = 0
        while i < len(spans) - 1:
            s1, s2 = spans[i], spans[i + 1]
            if (
                is_bold_like(s1) and
                is_bold_like(s2) and
                s1["y"] < top_limit and
                abs(s2["size"] - s1["size"]) < 0.1 and
                s1["font"] == s2["font"]
            ):
                line_spacing = s2["y"] - s1["y"]
                if line_spacing < 1.8 * s1["size"]:
                    title_lines = [s1]
                    while (
                        i + 1 < len(spans)
                        and is_bold_like(spans[i + 1])
                        and abs(spans[i + 1]["size"] - title_lines[-1]["size"]) < 0.1
                        and spans[i + 1]["y"] - title_lines[-1]["y"] < 1.8 * spans[i + 1]["size"]
                    ):
                        title_lines.append(spans[i + 1])
                        i += 1
                    title_text = " ".join([s["text"] for s in title_lines])
                    if title_text.strip():
                        return title_text.strip()
            i += 1

    # Final fallback to metadata
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title:
        return meta_title

    return "Untitled Document"


def ocr_img_block(img_bytes):
    return ocr_text_from_image(img_bytes)

def is_heading_candidate(text, size, flags, y, prev_y):
    # Enhanced heading candidate logic:
    # Heading if starts with bullet AND ends with colon (strong heading)
    # OR meets previous heuristics

    text = text.strip()
    if not text:
        return False

    starts_with_bullet = bool(BULLET_PATTERN.match(text))
    ends_with_colon = text.endswith(":")

    if starts_with_bullet and ends_with_colon:
        return True  # strong heading

    # Else apply previous heuristics
    if not (3 <= len(text) <= 100):
        return False
    if not text[0].isupper():
        return False
    if text[-1] in '.?!' and len(text) > 30:
        return False
    if len(re.findall(r'\s', text)) > 12:
        return False
    if text.count(',') > 1:
        return False
    if prev_y is not None:
        if (y - prev_y) < 4:
            return False
    return True

def cluster_sizes(sizes, threshold=0.7):
    sizes = sorted(sizes, reverse=True)
    clusters = []
    while sizes:
        base = sizes[0]
        cluster = [s for s in sizes if abs(s - base) <= threshold]
        clusters.append(sum(cluster)/len(cluster))
        sizes = [s for s in sizes if abs(s - base) > threshold]
    return clusters

def cluster_indentation_values(indent_list, threshold=5):
    indent_list = sorted(indent_list)
    clusters = []
    current_cluster = [indent_list[0]]
    for indent in indent_list[1:]:
        if abs(indent - current_cluster[-1]) <= threshold:
            current_cluster.append(indent)
        else:
            clusters.append(current_cluster)
            current_cluster = [indent]
    clusters.append(current_cluster)
    return [sum(c)/len(c) for c in clusters]

def assign_levels_by_size_and_indent(candidates, top_font_sizes, indent_threshold=5):
    levels = ["H1", "H2", "H3"]
    groups = []
    for size in top_font_sizes:
        filtered = [c for c in candidates if abs(c["size"] - size) < 0.01]
        if filtered:
            groups.append((size, filtered))

    outline = []
    for i, (size, group) in enumerate(groups):
        unique_indents = sorted(set(round(g["x"], 2) for g in group))
        clustered_indents = cluster_indentation_values(unique_indents, threshold=indent_threshold)
        # Debug: uncomment to see indent clusters
        # print(f"DEBUG: Font size {size:.2f}, indentation clusters: {clustered_indents}")

        indent_to_level = {}
        for j, indent in enumerate(clustered_indents):
            level_idx = min(i + j, len(levels) - 1)
            indent_to_level[indent] = levels[level_idx]

        for cand in group:
            closest_indent = min(clustered_indents, key=lambda ci: abs(ci - round(cand["x"], 2)))
            level = indent_to_level[closest_indent]
            outline.append({
                "level": level,
                "text": cand["text"],
                "page": cand["page"],
                "y": cand["y"],         # THIS LINE WAS ADDED/MODIFIED
                "x": cand["x"],         # THIS LINE WAS ADDED/MODIFIED
                "bbox": cand.get("bbox") # THIS LINE WAS ADDED/MODIFIED
            })
    return outline

def merge_consecutive_spans_with_bullets(spans, max_vertical_gap=12, max_indent_diff=10, max_font_size_diff=1):
    if not spans:
        return []

    spans = sorted(spans, key=lambda s: (s["page"], s["y"], s["x"]))
    merged = []
    buffer = [spans[0]]

    for current in spans[1:]:
        prev = buffer[-1]

        starts_with_bullet = bool(BULLET_PATTERN.match(current["text"]))

        same_page = current["page"] == prev["page"]
        vertical_close = abs(current["y"] - prev["y"]) <= max_vertical_gap
        indent_close = abs(current["x"] - prev["x"]) <= max_indent_diff
        font_size_close = abs(current["size"] - prev["size"]) <= max_font_size_diff

        # Only split if bullet AND line does NOT end with colon (to preserve bullet+colon headings)
        if (starts_with_bullet and not current["text"].endswith(":")) or not (same_page and vertical_close and indent_close and font_size_close):
            merged_text = " ".join(b["text"] for b in buffer)
            merged_span = buffer[0].copy()
            merged_span["text"] = merged_text
            merged.append(merged_span)
            buffer = [current]
        else:
            buffer.append(current)

    if buffer:
        merged_text = " ".join(b["text"] for b in buffer)
        merged_span = buffer[0].copy()
        merged_span["text"] = merged_text
        merged.append(merged_span)
    return merged

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    all_spans = []

    for pn in range(len(doc)):
        page = doc[pn]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get('type', 0) == 1 and 'image' in block:
                # Handle OCR for image blocks
                img_info = block['image']
                if isinstance(img_info, dict) and 'xref' in img_info:
                    xref = img_info['xref']
                else:
                    xref = img_info
                if isinstance(xref, int):
                    img = doc.extract_image(xref)
                    text = ocr_text_from_image(img['image'])
                    for line in text.split("\n"):
                        line_text = line.strip()
                        if line_text:
                            all_spans.append({
                                "text": line_text,
                                "size": 12, # A default size for OCR text
                                "flags": 0,
                                "page": pn + 1,
                                "y": block["bbox"][1] if block["bbox"] else 0,
                                "x": block["bbox"][0] if block["bbox"] else 0,
                                "bbox": block["bbox"] # Include block bbox for OCR text
                            })
            else:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        if span_text:
                            all_spans.append({
                                "text": span_text,
                                "size": span["size"],
                                "flags": span["flags"],
                                "font": span.get("font", ""),
                                "page": pn + 1,
                                "y": span["bbox"][1],
                                "x": span["bbox"][0],
                                "bbox": span["bbox"] # Keep bbox for later text block mapping
                            })

    merged_spans = merge_consecutive_spans_with_bullets(all_spans)

    spans_by_page = {}
    for s in merged_spans:
        spans_by_page.setdefault(s["page"], []).append(s)

    heading_candidates = []
    for page_num, spans in spans_by_page.items():
        prev_y = None
        for s in spans:
            if s["size"] != -1:
                if is_heading_candidate(
                    s["text"],
                    s.get("size", 0),
                    s.get("flags", 0),
                    s.get("y", 0),
                    prev_y
                ):
                    heading_candidates.append(s)
                    prev_y = s.get("y", 0)

    sizes = [s["size"] for s in heading_candidates if s["size"] > 1]
    if not sizes:
        sizes = [10, 12, 14, 16, 18, 20]

    uniq_sizes = sorted(list(set(sizes)), reverse=True)
    clustered_sizes = cluster_sizes(uniq_sizes, threshold=0.7)
    
    top_font_sizes = sorted(clustered_sizes, reverse=True)[:3] 

    outline = assign_levels_by_size_and_indent(heading_candidates, top_font_sizes)

    title = extract_pdf_title(doc)

    clean_outline = []
    for item in outline:
        if item["text"].strip().lower() != title.strip().lower():
            clean_outline.append(item)

    return {
        "title": title,
        "outline": clean_outline # This outline now includes 'y', 'x', 'bbox' for each heading
    }


if __name__ == "__main__":
    # This block is for standalone testing of pdf_processor.py
    # It will process PDFs from a specific collection folder and save their outlines.
    
    # Set the collection you want to test (e.g., "Collection 1", "Collection 2")
    collection_name_for_test = "Collection 1" 
    
    # Construct paths relative to your project's root (Challenge_1b)
    # os.path.dirname(__file__) gives '.../Challenge_1b/app'
    # os.path.dirname(os.path.dirname(__file__)) gives '.../Challenge_1b'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input PDF directory for this test
    input_pdfs_dir = os.path.join(project_root, collection_name_for_test, "PDFs")
    
    # Output directory for the generated outline JSONs from this test
    # (Creating a separate folder to avoid conflicts with challenge1b_output.json)
    output_outlines_dir = os.path.join(project_root, collection_name_for_test, "Outline_Outputs") 

    os.makedirs(output_outlines_dir, exist_ok=True)

    print(f"--- Running standalone test for pdf_processor.py ---")
    print(f"Looking for PDFs in: {input_pdfs_dir}")
    print(f"Saving outlines to: {output_outlines_dir}")

    if not os.path.exists(input_pdfs_dir):
        print(f"Error: Input PDF directory not found at {input_pdfs_dir}.")
        print(f"Please ensure '{collection_name_for_test}/PDFs' exists in your project root.")
    else:
        pdf_files = [f for f in os.listdir(input_pdfs_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {input_pdfs_dir}.")
        else:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_pdfs_dir, pdf_file)
                print(f"Processing {pdf_path} ...")
                try:
                    result = extract_outline(pdf_path) # Call the main function of this module
                    out_file = os.path.join(output_outlines_dir, os.path.splitext(pdf_file)[0] + ".json")
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"  --> Saved outline to {out_file}")
                except Exception as e:
                    print(f"Failed to process {pdf_file}: {e}")

    print("--- Standalone pdf_processor.py test finished. ---")