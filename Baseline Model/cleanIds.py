import json

input_path = "clinical_emotisupport_dataset.jsonl"
output_path = "clinical_emotisupport_dataset_numbered.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    for idx, line in enumerate(infile, start=1):
        record = json.loads(line)
        record["id"] = idx  # assign numeric ID
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Done. New file saved as:", output_path)
