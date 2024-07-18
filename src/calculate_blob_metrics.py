import pandas as pd
import logging
from collections import Counter
from fpdf import FPDF
from omegaconf import DictConfig
from utils.utils import read_yaml
from pathlib import Path
import os

log = logging.getLogger(__name__)

# Function to calculate image counts
def calculate_image_counts(df: pd.DataFrame) -> pd.DataFrame:

    image_counts = df[df['FileType'].isin(['jpg', 'png'])].groupby(['State', 'Month', 'Batch']).size().reset_index(name='ImageCount')
    log.info(f"Calculated image counts for {len(image_counts)} batches.")
    return image_counts

def calculate_average_image_counts(image_counts: pd.DataFrame) -> pd.DataFrame:
    # Calculates the average image counts grouped by state and month.
    average_image_counts = image_counts.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')
    log.info(f"Calculated average image counts for {len(average_image_counts)} state-month groups.")
    return average_image_counts




