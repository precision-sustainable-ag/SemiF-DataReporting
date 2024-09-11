import json
from pymongo import MongoClient
from pathlib import Path
from typing import Union, List
from omegaconf import DictConfig
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

class MongoDBDataLoader:
    """Class to handle loading JSON data into MongoDB from an NFS storage locker based on batches."""

    def __init__(self, cfg: DictConfig, data_type: str):
        """
        Initialize MongoDB connection and determine the collection and paths based on data_type.

        Args:
            cfg (DictConfig): OmegaConf DictConfig object containing the configuration settings.
            data_type (str): Type of data to load, either 'fullsized' or 'cutouts'.
        """
        self.cfg = cfg
        self.data_type = data_type
        
        # Set collection and paths based on the type of data
        if data_type == 'fullsized':
            self.collection_name = cfg.mongodb.fullsized_collection
            self.primary_nfs_root = Path(cfg.paths.longterm_images, "semifield-developed-images")
            self.secondary_nfs_root = Path(cfg.paths.GROW_DATA, "semifield-developed-images")
            self.csv_prefix = "semif_developed_batch_details"
        
        elif data_type == 'cutouts':
            self.collection_name = cfg.mongodb.cutout_collection
            self.primary_nfs_root = Path(cfg.paths.longterm_images, "semifield-cutouts")
            self.secondary_nfs_root = Path(cfg.paths.GROW_DATA, "semifield-cutouts")
            self.csv_prefix = "semif_cutouts_batch_details"
            
        else:
            raise ValueError(f"Invalid data_type '{data_type}' provided. Choose 'fullsized' or 'cutouts'.")

        self.client = MongoClient(f'mongodb://{cfg.mongodb.host}:{cfg.mongodb.port}/')
        self.db = self.client[cfg.mongodb.db]
        self.collection = self.db[self.collection_name]
        # Create a unique index on either 'cutout_id' or 'image_id field to enforce uniqueness and improve lookup performance
        self.create_id_index(data_type)

    def find_most_recent_csv(self, directory: Path) -> Path:
        """
        Finds the most recent CSV file in the directory matching 'prefix_YYYYMMDD.csv'.

        Args:
            directory (Path): The directory where CSV files are stored.

        Returns:
            Path: The path to the most recent CSV file.
        """
        files = list(directory.glob(f"{self.csv_prefix}_*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files with prefix '{self.csv_prefix}' found in {directory}")
        
        return max(files, key=lambda f: datetime.strptime(f.stem.split('_')[-1], '%Y%m%d'))

    def load_batches(self) -> List[str]:
        """
        Load batch names from the most recent CSV file.

        Returns:
            List[str]: A list of batch names.
        """
        processed_batch_path = self.find_most_recent_csv(Path(self.cfg.paths.tables_dir, self.data_type))
        df = pd.read_csv(processed_batch_path)
        batches = list(df[df["FormatType"] == "new"]["batch"])
        
        log.info(f"Loaded {len(batches)} batches from CSV file.")
        return batches

    def load_json_files_from_batches(self, batches: List[str]) -> None:
        """
        Load JSON files from batch directories in the NFS storage locker and insert them into MongoDB.
        If the batch is not found in the primary storage path, the script checks the alternative storage path.

        Args:
            batches (List[str]): List of batch directories to process.
        """
        primary_nfs_root_path = Path(self.primary_nfs_root)
        secondary_nfs_root_path = Path(self.secondary_nfs_root)

        for batch_name in tqdm(batches):
            batch_dir = primary_nfs_root_path / batch_name / ("metadata" if self.data_type == 'fullsized' else "")

            if batch_dir.is_dir():
                json_files = list(batch_dir.glob('*.json'))
                log.info(f"Processing batch '{batch_name}' in primary storage with {len(json_files)} JSON files.")
            else:
                batch_dir = secondary_nfs_root_path / batch_name / ("metadata" if self.data_type == 'fullsized' else "")
                if batch_dir.is_dir():
                    json_files = list(batch_dir.glob('*.json'))
                    log.info(f"Processing batch '{batch_name}' in alternative storage with {len(json_files)} JSON files.")
                else:
                    log.warning(f"Batch directory '{batch_name}' not found in either primary or alternative storage.")
                    continue

            for json_file_path in json_files:
                self.insert_data_from_file(json_file_path)

    def create_id_index(self, data_type: str) -> None:
        """Create a unique index on the 'cutout_id' field to enforce uniqueness and improve lookup performance."""
        try:
            index_value = "cutout_id" if data_type == 'cutouts' else "image_id"
            self.collection.create_index(index_value, unique=True)
            log.info(f"Unique index on '{index_value}' created successfully.")
        except Exception as e:
            log.error(f"Failed to create index on '{index_value}': {e}")

    def insert_data_from_file(self, json_file_path: Union[str, Path]) -> None:
        """
        Insert data from a JSON file into the MongoDB collection while avoiding duplicates.
        The uniqueness of the 'cutout_id' or 'image_id' field is enforced via the unique index.
        """
        try:
            with open(json_file_path) as file:
                data = json.load(file)

            # Determine the unique field based on the data type
            unique_field = "cutout_id" if self.data_type == 'cutouts' else "image_id"

            # Handle single document insertion
            if unique_field in data:
                try:
                    self.collection.insert_one(data)
                except Exception as e:
                    log.warning(f"Duplicate document with {unique_field} {data[unique_field]} found, skipping.")
            else:
                log.warning(f"Document missing '{unique_field}' field in {json_file_path}, skipping.")

        except Exception as e:
            log.error(f"Failed to insert data from {json_file_path}: {e}")


def main(cfg: DictConfig) -> None:
    
    # Initialize the loader with the MongoDB connection and the data type
    print(cfg.json_to_mongo.data_types)
    for data_type in cfg.json_to_mongo.data_types:
        # Initialize the loader with the MongoDB connection and the data type
        data_loader = MongoDBDataLoader(cfg, data_type)

        # Load batch names from the CSV file
        batch_names = data_loader.load_batches()

        # Load and insert JSON files from the batch directories
        data_loader.load_json_files_from_batches(batch_names)