import pandas as pd
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Type, Tuple
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from utils.utils import TqdmLoggingHandler

# Set up logging to integrate with tqdm
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
tqdm_handler = TqdmLoggingHandler()
log.addHandler(tqdm_handler)

class BatchValidator:
    def __init__(self):
        self.pattern: str = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'

    def is_valid_batch_name(self, batch_name: str) -> bool:
        """
        Validate the batch name against the naming convention pattern.
        """
        return re.match(self.pattern, batch_name) is not None

class BaseBatchChecker:
    def __init__(self, batch_path: Path, species_info: Dict[int, str]):
        self.batch_path: Path = batch_path
        self.validator: BatchValidator = BatchValidator()
        self.species_info = species_info

    def determine_format_type(self, metadata_path: Path, key: str) -> str:
        """
        Determine the format type of the metadata JSON file using a specific key.
        """
        if metadata_path.exists():
            for json_file in metadata_path.glob('*.json'):
                try:
                    with open(json_file, 'r') as file:
                        metadata = json.load(file)
                    if key in metadata:
                        return "new"
                    else:
                        return "old"
                except Exception as e:
                    log.error(f"Error reading JSON file {json_file}: {e}")
                break
        return "unknown"

    def get_annotation_data(self, metadata_path: Path, format_type: str) -> Tuple[Counter, Counter]:
        """
        Get the count of annotations or bboxes and their class IDs from the JSON metadata files based on the format type.
        """
        class_counter = Counter()
        primary_counter = Counter()
        if metadata_path.exists():
            for json_file in metadata_path.glob('*.json'):
                try:
                    with open(json_file, 'r') as file:
                        metadata = json.load(file)
                    if format_type == "new":
                        annotations = metadata.get("annotations", [])
                        for annotation in annotations:
                            class_id = annotation.get("category_class_id")
                            class_name = self.species_info.get(class_id, "error parsing common name in new format")
                            class_counter[class_name] += 1
                            if annotation.get("is_primary"):
                                primary_counter[class_name] += 1
                    elif format_type == "old":
                        bboxes = metadata.get("bboxes", [])
                        for bbox in bboxes:
                            cls = bbox.get("cls")
                            if isinstance(cls, dict):
                                class_id = cls.get("class_id")
                            else:
                                class_id = 27  # Default for "plant"
                            class_name = self.species_info.get(class_id, "error parsing common name in old format")
                            class_counter[class_name] += 1
                            if bbox.get("is_primary"):
                                primary_counter[class_name] += 1
                except Exception as e:
                    log.error(f"Error reading JSON file {json_file}: {e}")
        
        return class_counter, primary_counter

    def check_batch(self) -> Optional[Dict[str, any]]:
        """
        Check the batch for format type, annotation count, and class IDs.
        """
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.info(f"Valid batch found: {self.batch_path.name}")
            format_type = self.determine_format_type(self.batch_path / "metadata", "annotations")
            if format_type != "unknown":
                class_counter, primary_counter = self.get_annotation_data(self.batch_path / "metadata", format_type)
                image_count = self.get_image_count()

                batch_info = {
                    "batch": self.batch_path.name,
                    "FormatType": format_type,
                    "ImageCount": image_count,
                }

                # Add class counters to batch_info
                for class_name, count in class_counter.items():
                    batch_info[f"{class_name}_count"] = count

                # Add primary counters to batch_info
                for class_name, count in primary_counter.items():
                    batch_info[f"{class_name}_primary_count"] = count

                return batch_info
            else:
                return None
        else:
            log.warning(f"Ignoring invalid batch: {self.batch_path.name}")
        return None
    
    def get_image_count(self) -> int:
        """
        Get the count of .jpg images in the images subfolder.
        """
        images_path = self.batch_path / "images"
        return len(list(images_path.glob('*.jpg'))) if images_path.exists() else 0

class BatchReport:
    def __init__(self, paths: List[Path], checker_class: Type[BaseBatchChecker], species_info: Dict[int, str]):
        self.paths: List[Path] = paths
        self.checker_class = checker_class
        self.species_info = species_info

    def generate_report(self) -> List[Dict[str, any]]:
        """
        Generate a report by checking all batches in the provided paths.
        """
        all_batches_details = []
        for path in self.paths:
            base_path = Path(path)
            log.info(f"Checking path: {base_path}")
            batches = [x for x in base_path.glob('*')]
            for batch in tqdm(batches, desc=f"Processing batches in {base_path}", total=len(batches), leave=True, dynamic_ncols=True):
                if batch.is_dir():
                    checker = self.checker_class(batch, self.species_info)
                    batch_info = checker.check_batch()
                    if batch_info:
                        all_batches_details.append(batch_info)
        return all_batches_details

    def create_dataframe(self, batch_details: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Create a DataFrame from the batch details.
        """
        df = pd.DataFrame(batch_details).fillna(0)
        df.columns = df.columns.astype(str)  # Ensure all column names are strings
        return df

    def save_report(self, cfg: DictConfig, batch_details: List[Dict[str, any]], filename_prefix: str) -> None:
        """
        Save the report to a CSV file with a timestamp.
        """
        df = self.create_dataframe(batch_details)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_save_dir = Path(cfg.paths.data_dir, "species_distribution")
        csv_save_dir.mkdir(exist_ok=True, parents=True)
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_save_dir / csv_filename, index=False)
        log.info(f"Report saved to {csv_filename}")
        print(df)

def load_species_info(species_info_path: Path) -> Dict[int, str]:
    """
    Load the species information from the given JSON file.
    """
    with open(species_info_path, 'r') as file:
        species_info = json.load(file)
    return {v["class_id"]: v["common_name"] for k, v in species_info["species"].items()}

def generate_and_save_report(cfg: DictConfig, paths: List[Path], checker_class: Type[BaseBatchChecker], species_info: Dict[int, str], filename_prefix: str) -> None:
    """
    Generate and save the species distribution report for the given paths and checker class.
    """
    report = BatchReport(paths, checker_class, species_info)
    batch_details = report.generate_report()
    report.save_report(cfg, batch_details, filename_prefix)

def main(cfg: DictConfig) -> None:
    """
    Main function to execute the BatchReportGenerator.
    """
    log.info(f"Starting {cfg.task}")
    species_info_path = Path(cfg.paths.species_info)
    species_info = load_species_info(species_info_path)

    # Define paths for batches with subdirectories
    paths_with_subdirectories = [
        Path(cfg.paths.longterm_images, "semifield-developed-images"),
        Path(cfg.paths.GROW_DATA, "semifield-developed-images"),
    ]

    # Generate and save species distribution report for batches with subdirectories
    generate_and_save_report(cfg, paths_with_subdirectories, BaseBatchChecker, species_info, "species_distribution")
