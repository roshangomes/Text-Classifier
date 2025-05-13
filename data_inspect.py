import sys
import os
import pandas as pd
import re

# Add the path to 'src' where 'utils.py' is located
sys.path.append(os.path.abspath('./src'))
# Import the utils.py file
from utils import download_images

# Function to read CSV file and extract unique units
def read_Csv(data_path):
    df = pd.read_csv(data_path)
    
    # Create a new column for units extraction
    df['units'] = df['entity_value'].apply(lambda x: extract_units(x))
    
    # Get all unique units
    unique_units = df['units'].unique()

    # Select at least one instance of each unit
    selected_rows = pd.DataFrame()
    for unit in unique_units:
        unit_rows = df[df['units'] == unit]
        selected_rows = pd.concat([selected_rows, unit_rows.head(1)])  # Keep one row per unit

    # Fill the rest of the rows (up to 1000 rows) from the remaining dataset
    remaining_rows = df[~df.index.isin(selected_rows.index)]  # Exclude already selected rows
    additional_rows_needed = 200 - len(selected_rows)
    additional_rows = remaining_rows.sample(n=additional_rows_needed, random_state=42)
    
    # Combine the selected rows and additional rows
    final_df = pd.concat([selected_rows, additional_rows]).reset_index(drop=True)
    
    return final_df

# Function to extract units from the 'entity_value' column
def extract_units(entity_value):
    # Regular expression to match both numeric and unit patterns
    match = re.search(r'(\d+\.?\d*)\s*(\w+)', str(entity_value))
    if match:
        return match.group(2)  # Returns the unit part
    return None  # Returns None if no unit is found

# Function to download images
def download(df, download_folder="train_images3"):
    # Loop through the image links and download the images
    i = 0
    for link in df["image_link"]:
        try:
            i += 1
            print(f"Downloading image {i}")
            download_images([link], download_folder, allow_multiprocessing=False)  # download_images function handles downloading
        except Exception as e:
            print(f"Failed to download {link}: {e}")

