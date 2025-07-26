import spacy
# Ensure you've downloaded a spaCy model during Docker build, e.g., 'en_core_web_sm'
# python -m spacy download en_core_web_sm

class NLPAnalyzer:
    def __init__(self, persona, job_to_be_done):
        self.persona = persona
        self.job_to_be_done = job_to_be_done
        try:
            self.nlp = spacy.load("en_core_web_sm") # Load small English model
        except OSError:
            print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' during Docker build.")
            # Fallback or raise error, depending on your strategy
            self.nlp = None 

        # Process persona and job-to-be-done for keywords/entities
        self.persona_doc = self.nlp(persona) if self.nlp else None
        self.job_doc = self.nlp(job_to_be_done) if self.nlp else None

        # Extract keywords/entities from persona and job for better matching
        self.persona_keywords = [token.lemma_ for token in self.persona_doc if not token.is_stop and not token.is_punct] if self.persona_doc else []
        self.job_keywords = [token.lemma_ for token in self.job_doc if not token.is_stop and not token.is_punct] if self.job_doc else []
        
        # Combine for overall relevance scoring
        self.combined_query_tokens = set(self.persona_keywords + self.job_keywords)

    def calculate_relevance(self, text):
        """Calculates relevance score of text against persona and job-to-be-done."""
        if not self.nlp:
            return 0.0 # Cannot calculate without NLP model

        text_doc = self.nlp(text)
        
        # Simple keyword matching
        matched_keywords = [token.lemma_ for token in text_doc if token.lemma_ in self.combined_query_tokens]
        
        # You could use more advanced techniques here:
        # - Cosine similarity of document embeddings (if using larger models like Sentence-BERT, check size constraint!)
        # - Entity recognition and matching
        # - Semantic similarity using spaCy's .similarity() (requires a larger model like en_core_web_md/lg, check size)
        
        score = len(matched_keywords) / len(self.combined_query_tokens) if self.combined_query_tokens else 0.0
        return score

    def analyze_documents(self, processed_pdfs):
        """Analyzes each document to find relevant sections."""
        extracted_sections = []
        for pdf_data in processed_pdfs:
            doc_filename = pdf_data["filename"]
            outline = pdf_data["outline"]
            
            # Iterate through the extracted outline (headings)
            for section in outline:
                section_text = section["text"]
                page_number = section["page"]
                
                # Get surrounding text for better context if possible (from full_text)
                # This needs careful implementation to extract the actual content of a section
                # For now, let's just use the heading text itself for relevance
                
                relevance_score = self.calculate_relevance(section_text)
                
                # You might want to define a threshold for "relevance"
                if relevance_score > 0.1: # Example threshold
                    extracted_sections.append({
                        "document": doc_filename,
                        "section_title": section_text,
                        "raw_relevance_score": relevance_score, # Store for ranking
                        "page_number": page_number
                    })
        return extracted_sections

    def perform_subsection_analysis(self, ranked_sections, processed_pdfs):
        """
        Performs a more granular analysis within relevant sections to extract refined text.
        This would involve extracting text that falls under the identified relevant headings.
        """
        subsection_analysis_results = []
        
        # Create a map for quick access to full PDF text
        pdf_full_texts = {pdf['filename']: pdf['full_text'] for pdf in processed_pdfs}

        for section in ranked_sections:
            doc_filename = section["document"]
            section_title = section["section_title"]
            page_number = section["page_number"]
            
            full_text_for_doc = pdf_full_texts.get(doc_filename, "")
            
            # Logic to extract the specific text corresponding to this section title
            # This is complex and requires good PDF parsing capabilities (from R1A)
            # You might need to know the start and end page/position of a section.
            
            # For simplicity, let's assume we can get text around the heading
            # A more robust solution would use the outline structure to define section boundaries
            refined_text_placeholder = f"Content related to '{section_title}' from page {page_number}."
            
            # Search for the section title in the full text and extract content after it
            # This is a basic text search; proper implementation needs precise text block location
            # based on your R1A's detailed output
            
            subsection_analysis_results.append({
                "document": doc_filename,
                "refined_text": refined_text_placeholder,
                "page_number": page_number
            })
        return subsection_analysis_results