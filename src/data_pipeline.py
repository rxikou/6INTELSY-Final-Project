import pandas as pd
import re
import os

# --- PART 1: TOP OF FILE ---
DATA_DIR = "data/"
RAW_FILES = {
    "train": "train.tsv",
    "val": "val.tsv",
    "test": "test.tsv"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = " ".join(text.split())
    return text

def map_binary_labels(label):
    fake_labels = ['pants-on-fire', 'false', 'barely-true']
    real_labels = ['half-true', 'mostly-true', 'true']
    if label in fake_labels: return 0
    if label in real_labels: return 1
    return None

def process_dataset():
    # --- PART 2: INSIDE THE FUNCTION ---
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job', 
        'state', 'party', 'barely_true_counts', 'false_counts', 
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]
    
    for split, filename in RAW_FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
            
        # Updated loading logic for .tsv files
        df = pd.read_csv(file_path, sep='\t', names=columns, quoting=3) 
        
        # Cleaning steps
        df = df.dropna(subset=['statement'])
        df['label'] = df['label'].apply(map_binary_labels)
        df['statement'] = df['statement'].apply(clean_text)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        # Save as CSV for the Model Engineer
        output_path = os.path.join(DATA_DIR, f"cleaned_{split}.csv")
        df[['statement', 'label']].to_csv(output_path, index=False)
        print(f"Successfully saved {output_path}")

if __name__ == "__main__":
    process_dataset()