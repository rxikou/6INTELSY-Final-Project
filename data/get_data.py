import os
import urllib.request
import zipfile
import pandas as pd

def download_and_save_data():
    print("Downloading the official LIAR dataset directly from UCSB...")
    
    # Define paths
    output_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(output_dir, "liar_dataset.zip")
    
    # Official dataset URL
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    
    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting and formatting files...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    # The raw LIAR dataset doesn't have column headers, so we map them manually
    columns = [
        "id", "label", "statement", "subject", "speaker", "job_title", 
        "state", "party", "barely_true_counts", "false_counts", 
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    
    # Convert the extracted TSV files to clean CSVs
    splits = ['train', 'valid', 'test']
    for split in splits:
        tsv_file = os.path.join(output_dir, f"{split}.tsv")
        # Rename 'valid' to 'val' to match our pipeline
        csv_name = 'val.csv' if split == 'valid' else f"{split}.csv"
        csv_file = os.path.join(output_dir, csv_name)
        
        # Read the raw TSV and save as a clean CSV
        df = pd.read_csv(tsv_file, sep='\t', names=columns, header=None)
        df.to_csv(csv_file, index=False)
        
        # Clean up the raw TSV
        os.remove(tsv_file)
        
    # Clean up the zip file and readme
    os.remove(zip_path)
    if os.path.exists(os.path.join(output_dir, "README")):
        os.remove(os.path.join(output_dir, "README"))
        
    print(f"Data successfully cleaned and saved to {output_dir}")

if __name__ == "__main__":
    download_and_save_data()