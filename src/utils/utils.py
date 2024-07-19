import yaml
import logging
from tqdm import tqdm
import re
from pathlib import Path
import pandas as pd

def read_yaml(path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")

class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def get_most_recent_csv(folder_path: Path) -> Path:
    """Get the most recent CSV file from the folder with the filename format: species_distribution_YYYYMMDD_HHMMSS.csv."""
    csv_files = folder_path.glob('species_distribution_*.csv')
    date_pattern = re.compile(r'species_distribution_(\d{8}_\d{6}).csv')
    
    most_recent_file = None
    most_recent_date = None
    
    for csv_file in csv_files:
        match = date_pattern.search(csv_file.name)
        if match:
            file_date_str = match.group(1)
            file_date = pd.to_datetime(file_date_str, format='%Y%m%d_%H%M%S')
            if most_recent_date is None or file_date > most_recent_date:
                most_recent_date = file_date
                most_recent_file = csv_file
    
    return most_recent_file