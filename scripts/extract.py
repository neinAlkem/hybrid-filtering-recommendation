import kagglehub
import os

def download_dataset():
    
    """
    The function `download_dataset` downloads a dataset from Kaggle and saves it to a specified
    directory.
    """
    
    raw_path = 'dataset/raw_data.csv'
    os.mkdir(raw_path, exist_ok=True)
    
    path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews", download_dir=raw_path)
    
    print("Path to dataset files:", path)
