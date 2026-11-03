import json
import os

def evaluate_metrics(evidence_dir="evidence"):
    events_count = 0
    for root, dirs, files in os.walk(evidence_dir):
        if "data.json" in files:
            events_count += 1
    print(f"Total Evidence Sets Evaluated: {events_count}")
    return {"total_events": events_count}

if __name__ == "__main__":
    evaluate_metrics()