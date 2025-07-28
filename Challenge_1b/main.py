import os
import sys
import json
from datetime import datetime
from nlp_analyzer import NLPAnalyzer

def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python main.py <collection_folder>")
        sys.exit(1)

    collection = sys.argv[1]
    input_dir = os.path.join(collection)
    pdf_dir_path = os.path.join(input_dir, "PDFs")

    input_json_path = os.path.join(input_dir, "challenge1b_input.json")
    output_json_path = os.path.join(input_dir, "challenge1b_output.json")  # ✅ Output directly in collection_1

    if not os.path.exists(input_json_path):
        print(f"❌ Input JSON not found at {input_json_path}")
        sys.exit(1)

    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    print(f"🚀 Analyzing collection: {collection}")

    analyzer = NLPAnalyzer(pdf_base_path=pdf_dir_path)
    results = analyzer.analyze(input_data)

    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": results.get("extracted_sections", []),
        "subsection_analysis": results.get("subsection_analysis", [])
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Output written to {output_json_path}")

if __name__ == "__main__":
    main()
