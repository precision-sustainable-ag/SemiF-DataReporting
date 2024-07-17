import pandas as pd
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
import re
from typing import List, Dict, Optional
from datetime import datetime

log = logging.getLogger(__name__)

class BatchValidator:
    def __init__(self):
        self.pattern: str = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'

    def is_valid_batch_name(self, batch_name: str) -> bool:
        """
        Validate the batch name against the naming convention pattern.
        """
        return re.match(self.pattern, batch_name) is not None

class SubfolderCounter:
    def __init__(self, batch_path: Path):
        self.batch_path: Path = batch_path
        self.images_path: Path = batch_path / "images"
        self.metadata_path: Path = batch_path / "metadata"
        self.meta_masks_path: Path = batch_path / "meta_masks" / "semantic_masks"

    def get_subfolder_counts(self) -> tuple[int, int, int]:
        """
        Count the number of items in the subfolders: images, metadata, and meta_masks.
        """
        images_count: int = len(list(self.images_path.glob('*.jpg'))) if self.images_path.exists() else 0
        metadata_count: int = len(list(self.metadata_path.glob('*.json'))) if self.metadata_path.exists() else 0
        meta_masks_count: int = len(list(self.meta_masks_path.glob('*.png'))) if self.meta_masks_path.exists() else 0
        return images_count, metadata_count, meta_masks_count

    def get_missing_files(self) -> Dict[str, List[str]]:
        """
        Get the list of missing image, JSON, or mask files.
        """
        missing_files = {
            "missing_images": [],
            "missing_metadata": [],
            "missing_meta_masks": []
        }

        if self.images_path.exists() and self.metadata_path.exists():
            image_names = {f.stem for f in self.images_path.glob('*.jpg')}
            json_names = {f.stem for f in self.metadata_path.glob('*.json')}
            missing_files["missing_images"] = list(json_names - image_names)
            missing_files["missing_metadata"] = list(image_names - json_names)

        if self.meta_masks_path.exists():
            image_names = {f.stem for f in self.images_path.glob('*.jpg')}
            mask_names = {f.stem for f in self.meta_masks_path.glob('*.png')}
            missing_files["missing_meta_masks"] = list(image_names - mask_names)

        return missing_files

class BatchChecker:
    def __init__(self, batch_path: Path):
        self.batch_path: Path = batch_path
        self.validator: BatchValidator = BatchValidator()
        self.counter: SubfolderCounter = SubfolderCounter(batch_path)

    def check_has_matching(self, images_count: int, metadata_count: int, meta_masks_count: int) -> bool:
        """
        Check if the counts in images, metadata, and meta_masks subfolders match.
        """
        return images_count == metadata_count == meta_masks_count

    def check_unprocessed(self, metadata_path: Path, meta_masks_path: Path) -> bool:
        """
        Check if either the metadata or meta_masks subfolder is missing.
        """
        return not metadata_path.exists() or not meta_masks_path.exists()

    def determine_format_type(self, metadata_path: Path) -> str:
        """
        Determine the format type of the metadata JSON file.
        """
        if metadata_path.exists():
            for json_file in metadata_path.glob('*.json'):
                try:
                    with open(json_file, 'r') as file:
                        metadata = json.load(file)
                    if "annotations" in metadata:
                        return "new"
                    else:
                        return "old"
                except Exception as e:
                    log.error(f"Error reading JSON file {json_file}: {e}")
                break
        return "unknown"

    def check_batch(self) -> Optional[Dict[str, any]]:
        """
        Check the batch for subfolder counts, matching counts, unprocessed status, format type, and missing files.
        """
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.info(f"Valid batch found: {self.batch_path.name}")
            images_count, metadata_count, meta_masks_count = self.counter.get_subfolder_counts()
            has_matching = self.check_has_matching(images_count, metadata_count, meta_masks_count)
            unprocessed = self.check_unprocessed(self.counter.metadata_path, self.counter.meta_masks_path)
            format_type = self.determine_format_type(self.counter.metadata_path)
            missing_files = self.counter.get_missing_files()

            batch_info = {
                "path": str(self.batch_path.parent.parent.name),
                "batch": self.batch_path.name,
                "images": images_count,
                "metadata": metadata_count,
                "meta_masks": meta_masks_count,
                "HasMatching": has_matching,
                "UnProcessed": unprocessed,
                "FormatType": format_type,
                "MissingImages": missing_files["missing_images"],
                "MissingMetadata": missing_files["missing_metadata"],
                "MissingMetaMasks": missing_files["missing_meta_masks"]
            }

            return batch_info
        else:
            log.info(f"Ignoring invalid batch: {self.batch_path.name}")
        return None

class BatchReport:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.config_paths()
        
    def config_paths(self):
        self.paths = [
            Path(self.cfg.paths.longterm_images, "semifield-developed-images"),
            Path(self.cfg.paths.GROW_DATA, "semifield-developed-images")
            ]

    def generate_report(self) -> List[Dict[str, any]]:
        """
        Generate a report by checking all batches in the provided paths.
        """
        all_batches_details = []
        for path in self.paths:
            base_path = Path(path)
            log.info(f"Checking path: {base_path}")
            for batch in base_path.glob('*'):
                if batch.is_dir():
                    checker = BatchChecker(batch)
                    batch_info = checker.check_batch()
                    if batch_info:
                        all_batches_details.append(batch_info)
        return all_batches_details

    def create_dataframe(self, batch_details: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Create a DataFrame from the batch details.
        """
        return pd.DataFrame(batch_details)
    
    def save_csv(self, df):
        save_dir = Path(self.cfg.paths.data_dir,"storage_lockers")
        save_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        save_name = f"batch_details_{timestamp}.csv"
        df.to_csv(Path(save_dir, save_name), index=False)

def main(cfg: DictConfig) -> None:
    """
    Main function to execute the BlobMetricExporter.
    """
    log.info(f"Starting {cfg.task}")
    
    report = BatchReport(cfg)
    batch_details = report.generate_report()
    df = report.create_dataframe(batch_details)
    
    # Save the DataFrame to a CSV file
    report.save_csv(df)
    
    # Display the DataFrame
    print(df.head())


