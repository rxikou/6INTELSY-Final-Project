import os
from datasets import load_dataset

def download_and_save_data():
    print("Downloading the LIAR dataset for Fake News Detection...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("liar")
    
    # Define output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save splits to CSV
    dataset['train'].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    dataset['validation'].to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    dataset['test'].to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data successfully saved to {output_dir}")

if __name__ == "__main__":
    download_and_save_data()
