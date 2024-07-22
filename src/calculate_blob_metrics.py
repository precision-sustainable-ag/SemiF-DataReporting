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
def compute_matching( df: pd.DataFrame, matching_folders: str) -> dict:
    pd.set_option('display.max_colwidth', None)
    print(df.head(10))
     
    grouped_df=df[df['FolderName'].isin(matching_folders) & df['FileType'].isin(['json','jpg','png'])]
    grouped_df = grouped_df.groupby(['Batch','FileName'])
    log.info(f"Extracted DataFrame with {len(grouped_df)} rows for matching.")
    # Check for mismatch files
    batch_stat={x: {'images':0,'metadata':0,'meta_mask':0,'isMatching':True,'Missing':[]} for x in df.Batch.unique()}
    for key, item in grouped_df:
        if len(item) < 4:
            # assert len(item) == 0, f"{key} does not have any files."
            if len(item) == 0:
                log.info(f"{key} does not have any files.")

            log.info(f"{key} does not contain all reuqired files.")
            # Define the file types we are interested in
            file_types = {
                        'jpg': 'images',
                        'json': 'metadata',
                        'png': 'meta_mask'
                                        }

            for expected_file in file_types.keys():
                #update the file count
                if expected_file in list(item['FileType']):
                    batch_stat[key[0]][file_types[expected_file]] += 1
                else:
                    #create a list for missing files ignore unprocessed folders
                    batch_stat[key[0]]['Missing'].append(f"{item['FileName'].iloc[0]}.{expected_file}")
            batch_stat[key[0]]['isMatching'] = False
            
        else:
            batch_stat[key[0]]['images'] += 1
            batch_stat[key[0]]['metadata'] += 1
            batch_stat[key[0]]['meta_mask'] += 1
    
    return batch_stat





