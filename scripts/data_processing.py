import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_cleaning import clean_data

def process_and_save():
    # 1. Define file_path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_path = os.path.abspath(os.path.join(project_root,"data/raw/advertising.csv"))
    processed_dir = os.path.join(project_root,"data/processed")
    output_path = os.path.join(processed_dir, "_cleaned_advertising.csv")

    # 2. Check if raw_data_path exists:
    if not os.path.exists(raw_data_path):
        raise FileExistsError(f"Raw data file not found at {raw_data_path}")
    
    # 3. Load & clean
    print("Loading raw data...")
    df_raw = pd.read_csv(raw_data_path)
    
    print(f"Raw data size: {df_raw.shape}")
    df_clean = clean_data(df_raw)
    print(f"Cleaned data size: {df_clean.shape}")

    # 4. Save file 
    os.makedirs(processed_dir, exist_ok=True)
    
    # index=False to avoid saving extra serial number columns
    df_clean.to_csv(output_path, index=False)
    
    print(f"✅ Cleaned data saved at: {output_path}")

if __name__ == "__main__":
    process_and_save()
