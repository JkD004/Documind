import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdf_processor import extract_outline
from utils import clean_text
from ranker import SectionRanker


class NLPAnalyzer:
    def __init__(self, pdf_base_path=""):
        self.pdf_base_path = pdf_base_path
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=7000,
            ngram_range=(1, 3),
            min_df=1
        )
        self.ranker = SectionRanker()

    def _get_all_text_blocks_detailed(self, pdf_path):
        import fitz
        doc = fitz.open(pdf_path)
        all_blocks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            for block_dict in text_dict.get("blocks", []):
                if block_dict.get("type", 0) == 0:
                    block_text = ""
                    for line_dict in block_dict.get("lines", []):
                        for span_dict in line_dict.get("spans", []):
                            text = span_dict["text"].strip()
                            if text:
                                block_text += text + " "
                    if block_text.strip():
                        all_blocks.append({
                            "text": block_text.strip(),
                            "bbox": block_dict["bbox"],
                            "page": page_num + 1,
                            "y0": block_dict["bbox"][1]
                        })
        doc.close()
        return all_blocks

    def _map_outline_to_content(self, outline, doc_text_blocks_raw):
        sections_with_content = []

        sorted_outline = sorted(outline, key=lambda x: (x['page'], x.get('y', 0)))
        if not doc_text_blocks_raw:
            return []

        max_page = max(b['page'] for b in doc_text_blocks_raw)
        dummy_heading_y = max(b['y0'] for b in doc_text_blocks_raw if b['page'] == max_page) + 100
        sorted_outline.append({'text': '__END_OF_DOCUMENT__', 'page': max_page, 'y': dummy_heading_y, 'level': 'H1'})

        all_content_elements = []
        for block in doc_text_blocks_raw:
            all_content_elements.append({
                'type': 'text',
                'text': block['text'],
                'page': block['page'],
                'y': block['y0']
            })

        for heading in sorted_outline:
            all_content_elements.append({
                'type': 'heading',
                'text': heading['text'],
                'page': heading['page'],
                'y': heading['y'],
                'level': heading.get('level', 'H1')
            })

        all_content_elements.sort(key=lambda x: (x['page'], x['y']))

        current_heading_info = None
        current_section_full_text_parts = []

        for element in all_content_elements:
            if element['type'] == 'heading':
                if current_heading_info is not None and current_heading_info['text'] != '__END_OF_DOCUMENT__':
                    cleaned_title = self._clean_section_title(current_heading_info['text'])
                    if cleaned_title:
                        sections_with_content.append({
                            "section_title": cleaned_title,
                            "page_number": current_heading_info['page'],
                            "full_text_content": " ".join(current_section_full_text_parts).strip()
                        })
                current_heading_info = element
                current_section_full_text_parts = []
            elif element['type'] == 'text' and current_heading_info is not None:
                current_section_full_text_parts.append(element['text'])

        if sections_with_content and sections_with_content[-1]['section_title'] == '__END_OF_DOCUMENT__':
            sections_with_content.pop()

        return sections_with_content

    def _clean_section_title(self, title):
        if not title:
            return None
        title = title.strip()
        if title.endswith(":") or title.endswith("..."):
            title = title[:-1].strip()
        lower_title = title.lower()
        if lower_title in {"introduction", "conclusion", "summary"}:
            return None
        if len(title) > 100:
            title = title[:100].rsplit(' ', 1)[0] + "..."
        return title

    def _extract_refined_subsection(self, section_text, query_vector):
        paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        if not paragraphs:
            return section_text[:500] + "..." if len(section_text) > 500 else section_text

        paragraph_vectors = self.vectorizer.transform([clean_text(p) for p in paragraphs])
        if paragraph_vectors.shape[0] == 0:
            return section_text[:500] + "..." if len(section_text) > 500 else section_text

        paragraph_similarities = cosine_similarity(query_vector, paragraph_vectors)[0].tolist()
        scored_paragraphs = sorted(zip(paragraphs, paragraph_similarities), key=lambda x: x[1], reverse=True)

        refined_text_parts = []
        for p_text, p_score in scored_paragraphs:
            if p_score > 0.20 and len(p_text) > 80:
                refined_text_parts.append(p_text)
            if len(refined_text_parts) >= 3:
                break

        return " ".join(refined_text_parts) if refined_text_parts else (section_text[:500] + "..." if len(section_text) > 500 else section_text)

    def analyze(self, input_data):
        input_documents_info = input_data['documents']
        persona_role = input_data['persona']['role']
        job_task = input_data['job_to_be_done']['task']

        query_text = clean_text(persona_role + " " + job_task)
        if "travel" in persona_role.lower() or "trip" in job_task.lower():
            query_text += " itinerary beach nightlife adventure food cities packing friends group hotels"

        all_sections_for_ranking = []
        all_texts_for_vectorizer_fit = [query_text]

        for doc_info in input_documents_info:
            pdf_filename = doc_info['filename']
            pdf_path = os.path.join(self.pdf_base_path, pdf_filename)

            if not os.path.exists(pdf_path):
                print(f"⚠️ PDF not found at {pdf_path}. Skipping.")
                continue

            outline_data = extract_outline(pdf_path)
            doc_text_blocks_raw = self._get_all_text_blocks_detailed(pdf_path)
            sections_mapped_to_content = self._map_outline_to_content(outline_data['outline'], doc_text_blocks_raw)

            for section_data in sections_mapped_to_content:
                if section_data['full_text_content'].strip():
                    print(f"✅ Section found: {section_data['section_title']} in {pdf_filename}")
                    all_sections_for_ranking.append({
                        "document_filename": pdf_filename,
                        "section_title": section_data['section_title'],
                        "page_number": section_data['page_number'],
                        "full_text_content": section_data['full_text_content']
                    })
                    all_texts_for_vectorizer_fit.append(clean_text(section_data['full_text_content']))

        if not all_texts_for_vectorizer_fit:
            return {
                "extracted_sections": [],
                "subsection_analysis": []
            }

        self.vectorizer.fit(all_texts_for_vectorizer_fit)
        self.ranker.set_vectorizer(self.vectorizer)

        ranked_sections_output = self.ranker.rank_sections(
            all_sections_for_ranking,
            query_text,
            max_results=5,
            min_score_threshold=0.08
        )

        subsection_analysis = []
        top_ranked_sections_set = {(s['document'], s['section_title'], s['page_number']) for s in ranked_sections_output}

        for section in all_sections_for_ranking:
            key = (section['document_filename'], section['section_title'], section['page_number'])
            if key in top_ranked_sections_set:
                current_query_vector = self.vectorizer.transform([clean_text(query_text)])
                refined_text = self._extract_refined_subsection(section['full_text_content'], current_query_vector)
                if refined_text:
                    subsection_analysis.append({
                        "document": section['document_filename'],
                        "refined_text": refined_text,
                        "page_number": section['page_number']
                    })

        return {
            "extracted_sections": ranked_sections_output,
            "subsection_analysis": subsection_analysis
        }
