import json
import jsonschema
from pathlib import Path

schema_path = Path("sample_dataset/schema/output_schema.json")
outputs_path = Path("sample_dataset/outputs")

with open(schema_path) as f:
    schema = json.load(f)

for json_file in outputs_path.glob("*.json"):
    with open(json_file) as jf:
        try:
            data = json.load(jf)
            jsonschema.validate(instance=data, schema=schema)
            print(f"{json_file.name}: ✅ Valid")
        except jsonschema.ValidationError as ve:
            print(f"{json_file.name}: ❌ Invalid - {ve.message}")
