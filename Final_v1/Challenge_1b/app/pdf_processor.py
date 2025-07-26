import fitz # PyMuPDF
import json
import re
from collections import Counter
from pathlib import Path # Path might not be strictly needed here if passing str paths

# --- Heuristics & Configuration ---
TITLE_MIN_FONT_SIZE = 16
H1_MULTIPLIER = 1.5
H2_MULTIPLIER = 1.2
TITLE_LINE_TOLERANCE = 10


class PDFProcessor:
    @staticmethod
    def get_document_properties(blocks):
        font_sizes = [span['size'] for block in blocks if 'lines' in block for line in block['lines'] for span in line['spans']]
        return Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12

    @staticmethod
    def find_title(blocks_page_one, other_page_blocks):
        # Your existing find_title logic here
        repeating_text = {span['text'].strip() for block in other_page_blocks if 'lines' in block for line in block['lines'] for span in line['spans']}
        potential_lines = []

        for block in blocks_page_one:
            if 'lines' in block:
                for line in block['lines']:
                    line_text = " ".join(span['text'].strip() for span in line['spans'] if span['text'].strip())
                    if not line_text or line_text in repeating_text:
                        continue
                    span_sizes = [span['size'] for span in line['spans']]
                    if not span_sizes:
                        continue
                    avg_size = sum(span_sizes) / len(span_sizes)
                    if avg_size > TITLE_MIN_FONT_SIZE:
                        potential_lines.append({'text': line_text, 'size': avg_size, 'bbox': line['bbox']})

        if potential_lines:
            sorted_lines = sorted(potential_lines, key=lambda x: x['bbox'][1])
            grouped_titles, current_group = [], [sorted_lines[0]]

            for i in range(1, len(sorted_lines)):
                prev = current_group[-1]
                curr = sorted_lines[i]
                vertical_dist = curr['bbox'][1] - prev['bbox'][3]
                if vertical_dist < TITLE_LINE_TOLERANCE and abs(curr['size'] - prev['size']) < 1:
                    current_group.append(curr)
                else:
                    grouped_titles.append(current_group)
                    current_group = [curr]
            grouped_titles.append(current_group)

            title_candidates = [{
                'text': " ".join(line['text'] for line in group),
                'size': sum(line['size'] for line in group) / len(group),
                'y0': group[0]['bbox'][1]
            } for group in grouped_titles]

            return sorted(title_candidates, key=lambda x: (-x['size'], x['y0']))[0]['text']

        all_lines = [{'text': " ".join(span['text'].strip() for span in line['spans']), 'y0': line['bbox'][1]}
                      for block in blocks_page_one if 'lines' in block
                      for line in block['lines']
                      if any(span['text'].strip() for span in line['spans'])]

        return sorted(all_lines, key=lambda x: x['y0'])[0]['text'] if all_lines else "Untitled Document"

    @staticmethod
    def classify_heading(span, body_font_size):
        # Your existing classify_heading logic here
        span_size = span['size']
        span_text = span['text'].strip()

        if span_size <= body_font_size + 1:
            return None
        if re.match(r'^\d+\.\d+\.\d+', span_text): return "H3"
        if re.match(r'^\d+\.\d+', span_text): return "H2"
        if re.match(r'^\d+\.', span_text): return "H1"
        if re.search(r'^(Chapter|Appendix)\b', span_text, re.IGNORECASE): return "H1"
        if span_size > body_font_size * H1_MULTIPLIER: return "H1"
        if span_size > body_font_size * H2_MULTIPLIER: return "H2"
        return None

    @staticmethod
    def extract_structured_data(pdf_path: str) -> dict:
        """
        Extracts title, headings (H1, H2, H3 with level and page), and full text from a PDF.
        This function now incorporates your Round 1A logic.
        """
        doc = fitz.open(pdf_path)
        result = {"title": "Untitled Document", "outline": [], "full_text": ""}
        if len(doc) == 0:
            return result

        full_text_list = [] # To accumulate all text for 'full_text' output

        all_blocks_for_body_size = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks_on_page = page.get_text("dict")['blocks']
            all_blocks_for_body_size.extend(blocks_on_page)
            
            # Accumulate full text here
            full_text_list.append(page.get_text("text")) # Simple text extraction for full_text

        body_font_size = PDFProcessor.get_document_properties(all_blocks_for_body_size)

        first_page_blocks = doc.load_page(0).get_text("dict")['blocks']
        other_page_blocks = doc.load_page(1).get_text("dict")['blocks'] if len(doc) > 1 else []
        result['title'] = PDFProcessor.find_title(first_page_blocks, other_page_blocks)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")['blocks']
            
            # The original code skipped page 0 for outline extraction if doc > 1.
            # This might be an oversight if page 0 can contain H1/H2/H3.
            # For robustness, consider if you truly want to skip page 0 for headings.
            # I've removed that skip here for broader heading detection.
            
            for block in blocks:
                if 'lines' in block and block['lines']: # Ensure there are lines in the block
                    full_text_of_block = " ".join(span['text'].strip() for line in block['lines'] for span in line['spans'] if span['text'].strip())
                    
                    if not full_text_of_block:
                        continue
                    
                    first_span = block['lines'][0]['spans'][0]
                    heading_level = PDFProcessor.classify_heading(first_span, body_font_size)
                    
                    if heading_level:
                        result['outline'].append({
                            "level": heading_level,
                            "text": full_text_of_block,
                            "page": page_num + 1 # Page numbers are 1-indexed
                        })
        
        result['full_text'] = "\n".join(full_text_list)
        doc.close()
        return result

# The __main__ block for direct testing should be removed or commented out,
# as this file will be imported by main.py
# if __name__ == "__main__":
#     # This part is for standalone testing of pdf_processor.py
#     # For Challenge 1B, main.py will orchestrate the calls
#     input_dir = Path("sample_dataset/pdfs")
#     output_dir = Path("sample_dataset/outputs")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for pdf_file in input_dir.glob("*.pdf"):
#         try:
#             print(f"Processing: {pdf_file.name}")
#             outline_and_text = PDFProcessor.extract_structured_data(str(pdf_file))
#             output_file = output_dir / f"{pdf_file.stem}_1A_output.json" # Renamed to avoid confusion
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump(outline_and_text, f, indent=2, ensure_ascii=False)
#         except Exception as e:
#             print(f"Error processing {pdf_file.name}: {e}")