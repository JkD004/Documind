class Ranker:
    def rank(self, extracted_sections_raw):
        """
        Ranks the extracted sections by their relevance score.
        """
        # Sort sections by 'raw_relevance_score' in descending order
        sorted_sections = sorted(extracted_sections_raw, key=lambda x: x["raw_relevance_score"], reverse=True)
        
        # Assign importance_rank
        ranked_output = []
        for i, section in enumerate(sorted_sections):
            section["importance_rank"] = i + 1
            # Remove the temporary raw_relevance_score before final output
            del section["raw_relevance_score"] 
            ranked_output.append(section)
            
        return ranked_output