import pandas as pd
import logging
from collections import Counter
from fpdf import FPDF
from omegaconf import DictConfig
from utils.utils import read_yaml
from pathlib import Path

log = logging.getLogger(__name__)

# Function to calculate image counts
def calculate_image_counts(df: pd.DataFrame) -> pd.DataFrame:

    image_counts = df[df['FileType'].isin(['jpg', 'png'])].groupby(['State', 'Month', 'Batch']).size().reset_index(name='ImageCount')
    log.info(f"Calculated image counts for {len(image_counts)} batches.")
    return image_counts

# Function to calculate the average image counts grouped by state and month.
def calculate_average_image_counts(image_counts: pd.DataFrame) -> pd.DataFrame:
    average_image_counts = image_counts.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')
    log.info(f"Calculated average image counts for {len(average_image_counts)} state-month groups.")
    return average_image_counts

#Function to compare file type lenghts
def compute_matching( df: pd.DataFrame, matching_folders: str) -> None:
    pd.set_option('display.max_colwidth', None)
    print(df.head(10))

    grouped_df=df[df['FolderName'].isin(matching_folders) & df['FileType'].isin(['json','jpg','png'])]
    grouped_df = grouped_df.groupby(['Batch','FileName'])

    # Check for mismatch files
    for key, item in grouped_df:
        #uncomment for just one batch
        # if key[0] == 'MD_2022-06-27':
        if len(item) != 4:
            log.info(f"Found a batch that does not have all reuqired files.")
            print(grouped_df.get_group(key), "\n\n")
        else:
            continue
    # log.info(f"Found a batch that does not have all reuqired files.")
    return None





