from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

class SectionRanker:
    def __init__(self):
        self.vectorizer = None

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def rank_sections(self, sections_with_content, query_text, max_results=5, min_score_threshold=0.08):
        if not sections_with_content or not self.vectorizer:
            return []

        section_texts = [s['full_text_content'] for s in sections_with_content]
        query_vector = self.vectorizer.transform([clean_text(query_text)])
        section_vectors = self.vectorizer.transform(section_texts)

        similarities = cosine_similarity(query_vector, section_vectors)[0].tolist()

        scored_sections = []
        for i, section_data in enumerate(sections_with_content):
            title_lower = section_data['section_title'].lower()
            if title_lower in {"introduction", "conclusion", "summary"}:
                continue

            score = similarities[i]
            if score >= min_score_threshold:
                scored_sections.append({
                    "score": score,
                    **section_data
                })

        scored_sections.sort(key=lambda x: x['score'], reverse=True)
        top_sections = scored_sections[:max_results]

        ranked_output = []
        for rank, section in enumerate(top_sections, start=1):
            ranked_output.append({
                "document": section['document_filename'],
                "section_title": section['section_title'],
                "importance_rank": rank,
                "page_number": section['page_number']
            })

        return ranked_output
