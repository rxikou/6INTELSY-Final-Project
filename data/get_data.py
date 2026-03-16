import os
import urllib.request
import zipfile

def download_and_save_data():
    print("Downloading the official raw LIAR dataset directly from UCSB...")
    
    # Define paths
    output_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(output_dir, "liar_dataset.zip")
    
    # Official dataset URL
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    
    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting raw TSV files...")
    
    # Extract the files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
            
    # Rename 'valid.tsv' to 'val.tsv' to maintain pipeline consistency
    valid_tsv = os.path.join(output_dir, "valid.tsv")
    val_tsv = os.path.join(output_dir, "val.tsv")
    if os.path.exists(valid_tsv):
        os.rename(valid_tsv, val_tsv)
        
    # Clean up the zip file and remove the README
    os.remove(zip_path)
    readme_path = os.path.join(output_dir, "README")
    if os.path.exists(readme_path):
        os.remove(readme_path)
        
    print(f"Raw TSV data successfully extracted and cleaned up in {output_dir}")

if __name__ == "__main__":
    download_and_save_data()