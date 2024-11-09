import os
import glob
import pandas as pd

def load_and_merge_data(path_pattern, output_path):
    all_files = glob.glob(path_pattern)
    data_frames = [pd.read_csv(filename) for filename in all_files]
    merged_df = pd.concat(data_frames, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Data merged and saved to {output_path}")

def load_data(file_path):
    return pd.read_csv(file_path)
