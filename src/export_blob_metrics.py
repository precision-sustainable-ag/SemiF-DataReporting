#!/usr/bin/env python3
import logging
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import os
from utils.utils import read_yaml

log = logging.getLogger(__name__)


class ExporterBlobMetrics:
    """
    Exports blob metrics using Azure AzCopy list.

    Attributes:
    __auth_config_data: The authentication configuration data.
    output_dir: The output directory to save the blob list txt file.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BlobMetricExporter with configuration data.
        """
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.output_dir = cfg.paths.data_dir
        
    def run_azcopy_ls(self) -> None:
        """
        Runs the AzCopy command to list blobs and save the output to text files.
        """
        for k in tqdm(self.__auth_config_data["blobs"].keys()):
            if k == "account_url":
                continue
            else:
                url = self.__auth_config_data["blobs"][k]["url"]
                sas = self.__auth_config_data["blobs"][k]["sas_token"]
                n = 4 if k == "semifield-developed-images" else 2
                outputtxt = Path(self.output_dir, k + ".txt")
                # main azcopy list command
                cmd = f'./azcopy ls "{url + sas}" | cut -d/ -f 1-{n} | awk \'!a[$0]++\' > {outputtxt}'
                os.system(cmd)

    def organize_blob_list(self) -> None:
        """Organizes the list of blobs. (Implementation pending)"""
        pass

class CalculatorBlobMetrics:
    """
    Calculate blob metrics from the exported blob files.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BlobMetricsCalculator with configuration data.
        """
        self.output_dir = cfg.paths.data_dir
        self.report_dir = cfg.paths.report
        log.info("Initialized Exporting BlobMetrics with configuration data.")

    def extract_batches(self, lines: list[str]) -> list[tuple[str, str]]:
        """Function to extract batch names and file types"""
        batches = []
        for line in lines:
            # accounts for blank lines
            if not line.strip():
                continue
            # in some versions of azcopy
            if line.startswith('INFO: azcopy:'):
                continue
            
            parts = line.split('/')
            if line.startswith('INFO: '):
                
                batch = parts[0].replace('INFO: ', '').strip()
            else:
                batch = parts[0]
            
            full_filename = parts[-1].split(";")[0].strip()
            folder_name=parts[-2]
            
            if '.' in full_filename:
                file_type = full_filename.split('.')[-1]
                #hardcord to remove MD_2022-06-22
                if batch=='MD_2022-06-22' and file_type in ['json',]and len(full_filename.split('.'))>2:
                    continue
                    # rearrange=full_filename.split('.')[-3].split('_')
                    # file_name='MD_Row-'+rearrange[1]+'_'+rearrange[-1]
                # elif batch=='MD_2022-06-22' and file_type in ['jpg']:
                #     rearrange=full_filename.split('.')[-3].split('_')
                #     file_name='MD_Row-'+rearrange[1]+'_'+rearrange[-1]
                else:
                    file_name=full_filename.split('.')[-2]
            else:
                file_type = 'folder'
                file_name=full_filename

            batches.append((batch, folder_name, file_name, file_type))
        log.info(f"Extracted {len(batches)} batches.")
        return batches
    
    def remove_invalid_batches(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Removes invalid batches from DataFrame."""
        valid_pattern = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'
        invalid_batches = df[~df[column_name].str.contains(valid_pattern, regex=True)][column_name].unique()        
        filtered_df = df[df[column_name].str.contains(valid_pattern, regex=True)]
        
        log.info(f"Removed {len(invalid_batches)} unique batches due to invalid pattern.")
        return filtered_df

    def extract_month(self, batch_name: str) -> str:
        """ Extracts the month from the batch name."""
        parts = batch_name.split('_')
        month = parts[1][:7]
        return month

    def extract_state(self, batch_name: str) -> str:
        """Extracts the state from the batch name."""
        state = batch_name.split('_')[0]
        return state
    
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats and filters the data."""

        df = self.remove_invalid_batches(df, "Batch")
        # To avoid setting values on a copy of a slice from a DataFrame.
        df_copy = df.copy()
        # Apply the function to extract state and month
        df_copy['State'] = df['Batch'].apply(self.extract_state)
        df_copy['Month'] = df['Batch'].apply(self.extract_month)
        return df_copy

    def combine_data(self, image_counts: pd.DataFrame, average_image_counts: pd.DataFrame) -> pd.DataFrame:
        """ Combines image counts and average image counts into a single DataFrame."""

        result_df = pd.merge(image_counts, average_image_counts, on=['State', 'Month'])
        return result_df
        
    def calculate_image_counts(self,df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the image counts grouped by state, month, and batch."""

        image_counts = df[df['FileType'].isin(['jpg', 'png'])].groupby(['State', 'Month', 'Batch']).size().reset_index(name='ImageCount')
        log.info(f"Calculated image counts for {len(image_counts)} batches.")
        return image_counts

    def calculate_average_image_counts(self,image_counts: pd.DataFrame) -> pd.DataFrame:
        """Calculates the average image counts grouped by state and month."""

        average_image_counts = image_counts.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')
        log.info(f"Calculated average image counts for {len(average_image_counts)} state-month groups.")
        return average_image_counts

    def compute_matching(self,df: pd.DataFrame, matching_folders: str) -> pd.DataFrame:
        """Compares file type lengths."""

        pd.set_option('display.max_colwidth', None)
        print(df.head(10))
        file_types = {
                    'jpg': 'images',
                    'json': 'metadata',
                    'png': 'meta_mask'
                                            }
        
        grouped_df=df[df['FolderName'].isin(matching_folders) & df['FileType'].isin(file_types.keys())]
        grouped_df=grouped_df.groupby(["Batch", "FileType","FileName"]).size().reset_index(name='count')
        grouped_df=grouped_df.pivot_table(values='count', index=["Batch", "FileName"], columns='FileType', aggfunc='first').fillna(0).reset_index()
        grouped_df['all_files']=grouped_df[file_types.keys()].astype(float).sum(1)

        batch_df=grouped_df.groupby('Batch').agg({'jpg':'sum','json':'sum','png':'sum'}).reset_index()
        #hardcode for MD_2022-06-21
        batch_df.loc[batch_df.index[batch_df['Batch']=='MD_2022-06-21'],'json'] = batch_df.loc[batch_df.index[batch_df['Batch']=='MD_2022-06-21'],'jpg']

        # Check for mismatch files
        for key, item in batch_df.iterrows():
            batch_missing=[]
            #since the number of rows are small, we can conduct a manual check
            if item['jpg'] == item['jpg'] == item['png']/2:
                batch_df.loc[key, 'isMatching'] = 'True'
                batch_df.loc[key, 'Missing jpg'] = 0
                batch_df.loc[key, 'Missing json'] = 0
                batch_df.loc[key, 'Missing png'] = 0
                batch_df.at[key, 'Missing'] = []
            else:
                log.warn(f"{item['Batch']} is unprocessed.")
                batch_df.loc[key, 'isMatching'] = 'False'
                batch_df.loc[key, 'Missing jpg'] = max(item['jpg':'png'])-item['jpg']
                batch_df.loc[key, 'Missing png'] = max(item['jpg':'png'])-item['png']
                batch_df.loc[key, 'Missing json'] = max(item['jpg':'png'])-item['json']
                if item['json'] == item['png']==0.0:
                    batch_df.at[key, 'Missing'] = ['Unprocessed Batch']
                else:
                    log.warn(f"{item['Batch']} does not contain all reuqired files.")
                    # Define the file types we are interested in
                    for secondary_keys, expected_file in grouped_df[grouped_df['Batch']==item['Batch']].iterrows():
                        batch_missing.extend([f"{expected_file['FileName']}.{x[0]}" for x in expected_file['jpg':'png'].items() if x[1]==0.0])
                    
                    batch_df.at[key, 'Missing'] = batch_missing
                

        # uncomment to delete the key from the batch statistics and return the unprocessed batches
        # unprocessed_batches = []
        # for key, item in batch_df.iterrows():
            # item['Missing']==['Unprocessed Batch']
            # del batch_df[key] 
            # unprocessed_batches.append(item)  
            
        return batch_df


def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    log.info(f"Starting {cfg.task}")
    exporter = ExporterBlobMetrics(cfg)
    # exporter.run_azcopy_ls()
    log.info("Extracting data completed.")

    # Load the text file
    file_path = Path(cfg.paths.data_dir,'semifield-developed-images.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    log.info(f"Start calculating blob metrics with {len(lines)} of data.")
    calculator = CalculatorBlobMetrics(cfg)

    # Main execution
    batches = calculator.extract_batches(lines)

    df = pd.DataFrame(batches, columns=['Batch', 'FolderName', 'FileName', 'FileType'])
    log.info(f"Created DataFrame with {len(df)} rows.")

    df_filtered = calculator.format_data(df)

    # Calculate image counts and averages
    image_counts = calculator.calculate_image_counts(df_filtered)
    average_image_counts = calculator.calculate_average_image_counts(image_counts)

    #Compare file type lengths
    mismatch_statistics=calculator.compute_matching(df_filtered,cfg.matching_folders)

    #writing the mismatch statistics to a csv file for now
    mismatch_statistics.to_csv(Path(calculator.report_dir,'mismatch_statistics_record.csv'), sep=',', encoding='utf-8', index=False, header=True)

    log.info(f"{cfg.task} completed.")
