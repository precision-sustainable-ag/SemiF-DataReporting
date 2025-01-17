#!/usr/bin/env python3
import logging
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import os
from utils.utils import read_yaml
import shutil

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
        Args:
            cfg: DictConfig - The configuration data.
        """
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.output_dir = Path(cfg.paths.data_dir, "blob_containers")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
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
                if os.path.exists("./azcopy"):
                    cmd = f'./azcopy ls "{url + sas}" | cut -d/ -f 1-{n} | awk \'!a[$0]++\' > {outputtxt}'
                else:
                    cmd = f'azcopy ls "{url + sas}" | cut -d/ -f 1-{n} | awk \'!a[$0]++\' > {outputtxt}'
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
        args:
            cfg: The configuration data

        Arguments:
        output_dir: The output directory to save the blob list txt file.
        report_dir: The report directory to save the mismatch statistics.
        """
        self.output_dir = cfg.paths.data_dir
        self.report_dir = cfg.paths.report
        self.matching_folders = cfg.matching_folders
        log.info("Initialized Exporting BlobMetrics with configuration data.")

    def extract_batches(self, lines: list[str], txt_name:str) -> list[tuple[str, str]]:
        """Function to extract batch names and file types
        Args:
            lines: The list of lines from the blob list text file.
            file_name: A boolean to indicate if the blob is a cutout blob or not.
        Returns:
            The list of tuples containing the batch name file name and file type.
            """
        
        batches = []
        for line in lines:
            # accounts for blank lines
            if not line.strip():
                continue
            # in some versions of azcopy
            if line.startswith('INFO: azcopy'):
                continue
            
            parts = line.split('/')
            if line.startswith('INFO: '):
                
                batch = parts[0].replace('INFO: ', '').strip()
            else:
                batch = parts[0]
            
            full_filename = parts[-1].split(";")[0].strip()

            # Extract the folder name from the full filename
            if 'cutouts' in txt_name:
                if 'mask' in full_filename:
                    folder_name=parts[-1].split('_')[-3]
                else:
                    folder_name=parts[-1].split('_')[-2]
            else:
                folder_name=parts[-2]
            
            if '.' in full_filename:
                file_type = full_filename.split('.')[-1]
                #hardcode to ignore the json part of MD_2022-06-22 and 2022-06-21
                if batch=='MD_2022-06-22' and file_type in ['json',] and len(full_filename.split('.'))>2:
                    rearrange=full_filename.split('.')[-3].split('_')
                    file_name='MD_Row-'+rearrange[1]+'_'+rearrange[-1]
                if batch=='MD_2022-06-21' and file_type in ['json',] and len(full_filename.split('.'))>2:
                    continue    
                else:
                    file_name=full_filename.split('.')[-2]
            else:
                file_type = 'folder'
                file_name=full_filename
            batches.append((batch, folder_name, file_name, file_type))
        log.info(f"Extracted {len(batches)} batches.")
        return batches
    
    def remove_invalid_batches(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Removes invalid batches from DataFrame.
        Args:
            df: The DataFrame containing the batch data.
            column_name: The name of the column containing the batch names.
        Returns:
            The filtered DataFrame."""
        
        valid_pattern = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'
        invalid_batches = df[~df[column_name].str.contains(valid_pattern, regex=True)][column_name].unique()        
        filtered_df = df[df[column_name].str.contains(valid_pattern, regex=True)]
        
        log.info(f"Removed {len(invalid_batches)} unique batches due to invalid pattern.")
        return filtered_df

    def extract_month(self, batch_name: str) -> str:
        """ Extracts the month from the batch name.
        Args:
            batch_name: The name of the batch.
        Returns:
            The month extracted from the batch name."""
        parts = batch_name.split('_')
        month = parts[1][:7]
        return month

    def extract_state(self, batch_name: str) -> str:
        """Extracts the state from the batch name.
        Args:
            batch_name: The name of the batch.
        Returns:
            The state extracted from the batch name.
            """
        state = batch_name.split('_')[0]
        return state
    
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats and filters the data.
        Args:
            df: The DataFrame containing the batch data.
        Returns:
            The formatted DataFrame.
        """

        df = self.remove_invalid_batches(df, "Batch")
        # To avoid setting values on a copy of a slice from a DataFrame.
        df_copy = df.copy()
        # Apply the function to extract state and month
        df_copy['State'] = df['Batch'].apply(self.extract_state)
        df_copy['Month'] = df['Batch'].apply(self.extract_month)
        return df_copy

    def combine_data(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """ Combines image counts and average image counts into a single DataFrame.
        Args:
            df1: First DataFrame.
            df2: Second DataFrame.
        Returns:
            The combined DataFrame.
            """

        result_df = pd.merge(df1, df2, on=['State', 'Month'])
        return result_df
        
    def calculate_image_counts(self,df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the image counts grouped by state, month, and batch.
        Args:
            df: The DataFrame containing the batch data.
        Returns:
            The DataFrame containing the image counts."""

        image_counts = df[df['FileType'].isin(['jpg', 'png'])].groupby(['State', 'Month', 'Batch']).size().reset_index(name='ImageCount')
        log.info(f"Calculated image counts for {len(image_counts)} batches.")
        return image_counts

    def calculate_average_image_counts(self,image_counts: pd.DataFrame) -> pd.DataFrame:
        """Calculates the average image counts grouped by state and month.
        Args:
            image_counts: The DataFrame containing the image counts.
        Returns:
            The DataFrame containing the average image counts."""

        average_image_counts = image_counts.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')
        log.info(f"Calculated average image counts for {len(average_image_counts)} state-month groups.")
        return average_image_counts

    def compute_matching(self,df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compares file type lengths.
        Args:
            df: The DataFrame containing the batch data.
        Returns:
            The DataFrame containing the mismatch statistics.
            The DataFrame containing unprocessed batches.
        """
        
        # Filter the DataFrame to include only the matching folders and file types
        grouped_df=df[df['FolderName'].isin(self.matching_folders.keys()) & df['FileType'].isin(self.matching_folders.values())]
        
        # Group the DataFrame by Batch, FileType, and FileName and calculate the number of files
        grouped_df=grouped_df.groupby(["Batch", "FolderName","FileName"]).size().reset_index(name='count')
        
        # Pivot the DataFrame to have the file types as columns
        grouped_df=grouped_df.pivot_table(values='count', index=["Batch", "FileName"], columns='FolderName', aggfunc='first').fillna(0).reset_index()

        # Calculate the total number of files per one capture
        grouped_df['all_files']=grouped_df[self.matching_folders.keys()].astype(float).sum(1)
        # Calculate the total number of files per batch
        batch_df=grouped_df.groupby('Batch').agg({'images':'sum','metadata':'sum','instance_masks':'sum','semantic_masks':'sum'}).reset_index()
        
        # Check for mismatch files
        for key, item in batch_df.iterrows():
            batch_missing=[]
            #since the number of rows are small, we can conduct a manual check
            if item['metadata'] == item['images'] == item['semantic_masks'] == item['instance_masks']:
                batch_df.loc[key, 'isMatching'] = 'True'
                batch_df.loc[key, 'Missing images'] = 0
                batch_df.loc[key, 'Missing metadata'] = 0
                batch_df.loc[key, 'Missing instance_masks'] = 0
                batch_df.loc[key, 'Missing semantic_masks'] = 0
                batch_df.at[key, 'Missing'] = []
            else:
                batch_df.loc[key, 'isMatching'] = 'False'
                batch_df.loc[key, 'Missing images'] = max(item['images':'semantic_masks'])-item['images']
                batch_df.loc[key, 'Missing metadata'] = max(item['images':'semantic_masks'])-item['metadata']
                batch_df.loc[key, 'Missing instance_masks'] = max(item['images':'semantic_masks'])-item['instance_masks']
                batch_df.loc[key, 'Missing semantic_masks'] = max(item['images':'semantic_masks'])-item['semantic_masks']
                if item['metadata'] == item['instance_masks'] == item['semantic_masks']  ==0.0:
                    log.warn(f"{item['Batch']} is unprocessed.")
                    batch_df.at[key, 'Missing'] = ['Unprocessed Batch']
                else:
                    log.warn(f"{item['Batch']} does not contain all reuqired files.")
                    # Define the file types we are interested in
                    for secondary_keys, expected_file in grouped_df[grouped_df['Batch']==item['Batch']].iterrows():
                        batch_missing.extend([f"{expected_file['FileName']}.{self.matching_folders[x[0]]}" for x in expected_file['images':'semantic_masks'].items() if x[1]==0.0])
                    
                    batch_df.at[key, 'Missing'] = batch_missing
                

        # create a seperate list of fully unprocessed batches
        unprocessed_batches = []
        for key, item in batch_df.iterrows():
            if item['Missing']==['Unprocessed Batch']:
                unprocessed_batches.append(item) 
        unprocessed_batches=pd.DataFrame(unprocessed_batches)  

        log.info(f"Calculated mismatch statistics for {len(batch_df)} batches.")  
        return batch_df, unprocessed_batches

    def compare_cutout_blob(self,df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compares file type lengths.
        Args:
            df: The DataFrame containing the batch data.
        Returns:
            The DataFrame containing the match statistics.
            The DataFrame containing unprocessed batches.
        """

        # rename FolderName to TimeStamp for convienience
        df.rename(columns={'FolderName':'TimeStamp'}, inplace=True)
        # Filter the DataFrame to include only the matching folders and file types
        grouped_df = df[df['FileType'].isin(self.matching_folders.values())]
        
        #update the file type to include the mask files
        grouped_df.loc[grouped_df['FileName'].str.contains('mask'), 'FileType'] = 'mask'
        #remove the _mask from the file name
        grouped_df.loc[:,'FileName']=grouped_df['FileName'].map(lambda x: x.replace("_mask", "") if 'mask' in x else x)
        #number of the number of:masks (*_mask.png), cropouts (*.jpg), cutouts (*.png), metadata (*.json)
        batch_df=grouped_df.groupby(["Batch", "FileType"]).size().reset_index(name='count')
        # Pivot the DataFrame to have the file types as columns
        batch_df=batch_df.pivot_table(values='count', index="Batch", columns='FileType', aggfunc='first').fillna(0).reset_index()

        # Group the DataFrame by Batch, FileType, and FileName and calculate the number of files
        grouped_df=grouped_df.groupby(["Batch", "TimeStamp","FileName"])['FileType'].agg(lambda col: ','.join(col)).reset_index(name='TypeList')
        grouped_df['Count']=grouped_df['TypeList'].apply(lambda x: len(x.split(',')))
        
        # find the unmatched files
        unprocessed_df=grouped_df[~(grouped_df['Count'] == 4)]
        
        # Calculate the missing number of files for each batch
        n_missing_cropout=[1 if 'jpg' not in x else 0 for x in unprocessed_df['TypeList']]
        n_missing_cutouts=[1 if 'png' not in x else 0 for x in unprocessed_df['TypeList']]
        n_missing_masks=[1 if 'mask' not in x else 0 for x in unprocessed_df['TypeList']]
        n_missing_metadata=[1 if 'json' not in x else 0 for x in unprocessed_df['TypeList']]

        new_data = {'n_missing_cropout': n_missing_cropout, 'n_missing_cutouts': n_missing_cutouts, 'n_missing_masks': n_missing_masks, 'n_missing_metadata': n_missing_metadata}
        unprocessed_df = unprocessed_df.assign(**new_data)

        #drop temp columns
        unprocessed_df.drop(columns=['TypeList'], inplace=True)

        return batch_df, unprocessed_df
    
    def load_data(self, data_file: Path) -> pd.DataFrame:
        """Loads the data from the given path.
        Args:
            path: The path to the data file.
        Returns:
            The DataFrame containing the data.
            """
        # Load the text file
        file_path = Path(self.output_dir,data_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        log.info(f"Start calculating blob metrics with {len(lines)} of data.")

        #Extract the batches and create a DataFrame
        batches = self.extract_batches(lines,data_file)
        df = pd.DataFrame(batches, columns=['Batch', 'FolderName', 'FileName', 'FileType'])
        log.info(f"Created DataFrame with {len(df)} rows.")

        # Format the dataframe
        df_filtered = self.format_data(df)
        return df_filtered
    
    def save_data(self, df_stat: pd.DataFrame, df_unprocess: pd.DataFrame, file_names: list) -> None:
        """Saves the data to the given path.
        Args:
            df: The DataFrame containing the data.
            file_name: Names of the files to save the data.
            """
        save_csv_dir = Path(self.output_dir, "blob_containers")
        save_csv_dir.mkdir(exist_ok=True, parents=True)
        df_stat.to_csv(Path(save_csv_dir, file_names[0]), sep=',', encoding='utf-8', index=False, header=True)
        df_unprocess.to_csv(Path(save_csv_dir, file_names[1]), sep=',', encoding='utf-8', index=False, header=True)
        return None

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""

    log.info(f"Starting {cfg.task}")
    exporter = ExporterBlobMetrics(cfg)
    # exporter.run_azcopy_ls()
    log.info("Extracting data completed.")


    calculator = CalculatorBlobMetrics(cfg)
    df_developed=calculator.load_data('semifield-developed-images.txt')
    df_cutout=calculator.load_data('semifield-cutouts.txt')
    
    # Calculate image counts and averages
    image_counts = calculator.calculate_image_counts(df_developed)
    average_image_counts = calculator.calculate_average_image_counts(image_counts)

    #Compare file type lengths
    mismatch_statistics_cutout, unprocessed_batches_cutout=calculator.compare_cutout_blob(df_cutout)
    mismatch_statistics_developed, unprocessed_batches_developed=calculator.compute_matching(df_developed)
    

    #writing the mismatch statistics to a csv file for now
    
    calculator.save_data(mismatch_statistics_developed, unprocessed_batches_developed, ['mismatch_statistics_developed.csv','unprocessed_batches_developed.csv'])
    calculator.save_data(mismatch_statistics_cutout, unprocessed_batches_cutout, ['statistics_cutout.csv','batches_wih_missed_files_cutout.csv'])
    
    log.info(f"{cfg.task} completed.")