import json
import re
import os
from collections import Counter

# Use path relative to this script file to avoid CWD errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'dataset_refined_v6_reindexed.jsonl')

# Artifact patterns to detect
ARTIFACT_PATTERNS = [
    r"^Here are the main points:\s*-?",
    r"^Quick question:\s*",
    r"^To clarify,?\s*",
    r"^Look,?\s*",
    r"^Regarding\s+my\s+last\s+call,?\s*",
    r"^Need clarification:\s*",
    r"^What happened:\s*",
    r"^The nurse said,\s*\"",
]

def check_quality(file_path):
    issues = {
        "logic_conflict": 0,    # Satisfaction + Negative
        "label_flooding": 0,    # > 3 labels
        "has_artifact": 0,      # Text starts with artifact
        "short_text": 0,        # < 5 words
        "total_records": 0
    }
    
    # Specific IDs for debugging
    conflict_ids = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                issues["total_records"] += 1
                rec = json.loads(line)
                
                # Check 1: Logic Conflict
                emotions = rec.get("emotions", {})
                # Normalize to 1/0
                emotions = {k: (1 if v else 0) for k, v in emotions.items()}
                
                negatives = ["anger", "frustration", "disappointment", "anxiety"]
                has_substantive_negative = any(emotions.get(e, 0) == 1 for e in negatives)
                is_satisfied = emotions.get("satisfaction", 0) == 1
                
                if has_substantive_negative and is_satisfied:
                    issues["logic_conflict"] += 1
                    conflict_ids.append(rec.get("id"))

                # Check 2: Label Flooding
                active_count = sum(emotions.values())
                if active_count > 3:
                    issues["label_flooding"] += 1

                # Check 3: Artifacts
                text = rec.get("text", "")
                for pat in ARTIFACT_PATTERNS:
                    if re.search(pat, text, re.IGNORECASE):
                        issues["has_artifact"] += 1
                        break
                
                # Check 4: Short texts
                if len(text.split()) < 5:
                    issues["short_text"] += 1

        print(json.dumps(issues, indent=2))
        print(f"\nSample IDs with Logic Conflicts (Satisfaction + Negative): {conflict_ids[:5]}...")

    except FileNotFoundError:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    check_quality(INPUT_FILE)
