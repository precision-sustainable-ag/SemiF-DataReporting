import pandas as pd
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple, Type
from datetime import datetime
from tqdm import tqdm
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

class BaseCounter:
    def __init__(self, batch_path: Path):
        self.batch_path: Path = batch_path

    def get_missing_files(self) -> Dict[str, List[str]]:
        """
        Abstract method to get the list of missing files.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

class SubfolderCounter(BaseCounter):
    def __init__(self, batch_path: Path):
        super().__init__(batch_path)
        self.images_path: Path = batch_path / "images"
        self.metadata_path: Path = batch_path / "metadata"
        self.meta_masks_path: Path = batch_path / "meta_masks" / "semantic_masks"

    def get_subfolder_counts(self) -> Tuple[int, int, int]:
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

class CutoutCounter(BaseCounter):
    def get_cutout_counts(self) -> Tuple[int, int, int, int]:
        """
        Count the number of items in the batch folder: .jpg, .png, .json, and _mask.png files.
        """
        jpg_count: int = len(list(self.batch_path.glob('*.jpg')))
        png_count: int = len(list(self.batch_path.glob('*.png'))) - len(list(self.batch_path.glob('*_mask.png')))
        json_count: int = len(list(self.batch_path.glob('*.json')))
        mask_count: int = len(list(self.batch_path.glob('*_mask.png')))
        return jpg_count, png_count, json_count, mask_count

    def get_missing_files(self) -> Dict[str, List[str]]:
        """
        Get the list of missing .jpg, .png, .json, or _mask.png files.
        """
        log.info("Getting all files in batch path...")
        all_files = list(self.batch_path.glob('*'))

        log.info("Processing file extensions...")
        stems = set(f.stem for f in all_files if f.suffix in {'.jpg', '.png', '.json', '_mask.png'})
        jpg_stems = {f.stem for f in all_files if f.suffix == '.jpg'}
        png_stems = {f.stem for f in all_files if f.suffix == '.png'}
        json_stems = {f.stem for f in all_files if f.suffix == '.json'}
        mask_stems = {f.stem[:-5] for f in all_files if f.suffix == '.png' and f.stem.endswith('_mask')}

        missing_files = {
            "missing_jpg": list(stems - jpg_stems),
            "missing_png": list(stems - png_stems),
            "missing_json": list(stems - json_stems),
            "missing_mask": list(stems - mask_stems)
        }

        return missing_files

class BaseBatchChecker:
    def __init__(self, batch_path: Path):
        self.batch_path: Path = batch_path
        self.validator: BatchValidator = BatchValidator()

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

    def check_batch(self) -> Optional[Dict[str, any]]:
        """
        Abstract method to check the batch.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

class SemifieldDevelopedBatchChecker(BaseBatchChecker):
    def __init__(self, batch_path: Path):
        super().__init__(batch_path)
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

    def check_batch(self) -> Optional[Dict[str, any]]:
        """
        Check the batch for subfolder counts, matching counts, unprocessed status, format type, and missing files.
        """
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.info(f"Valid batch found: {self.batch_path.name}")
            images_count, metadata_count, meta_masks_count = self.counter.get_subfolder_counts()
            has_matching = self.check_has_matching(images_count, metadata_count, meta_masks_count)
            unprocessed = self.check_unprocessed(self.counter.metadata_path, self.counter.meta_masks_path)
            format_type = self.determine_format_type(self.counter.metadata_path, "annotations")
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
            log.warning(f"Ignoring invalid batch: {self.batch_path.name}")
        return None

class SemifieldCutoutBatchChecker(BaseBatchChecker):
    def __init__(self, batch_path: Path):
        super().__init__(batch_path)
        self.counter: CutoutCounter = CutoutCounter(batch_path)

    def check_has_matching(self, jpg_count: int, png_count: int, json_count: int, mask_count: int) -> bool:
        """
        Check if the counts of .jpg, .png, .json, and _mask.png files match.
        """
        return jpg_count == png_count == json_count == mask_count

    def check_batch(self) -> Optional[Dict[str, any]]:
        """
        Check the batch for cutout counts, matching counts, and missing files.
        """
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.info(f"Valid batch found: {self.batch_path.name}")
            jpg_count, png_count, json_count, mask_count = self.counter.get_cutout_counts()
            has_matching = self.check_has_matching(jpg_count, png_count, json_count, mask_count)
            format_type = self.determine_format_type(self.batch_path, "category")
            missing_files = self.counter.get_missing_files()

            batch_info = {
                "path": str(self.batch_path.parent.parent.name),
                "batch": self.batch_path.name,
                "jpg_count": jpg_count,
                "png_count": png_count,
                "json_count": json_count,
                "mask_count": mask_count,
                "HasMatching": has_matching,
                "FormatType": format_type,
                "MissingJpg": missing_files["missing_jpg"],
                "MissingPng": missing_files["missing_png"],
                "MissingJson": missing_files["missing_json"],
                "MissingMask": missing_files["missing_mask"]
            }

            return batch_info
        else:
            log.warning(f"Ignoring invalid batch: {self.batch_path.name}")
        return None

class BatchReport:
    def __init__(self, paths: List[Path], checker_class: BaseBatchChecker):
        self.paths: List[Path] = paths
        self.checker_class = checker_class

    def generate_report(self) -> List[Dict[str, any]]:
        """
        Generate a report by checking all batches in the provided paths.
        """
        all_batches_details = []
        for path in self.paths:
            base_path = Path(path)
            log.info(f"Checking path: {base_path}")
            batches = [x for x in base_path.glob('*')]
            # for batch in base_path.glob('*'):
            for batch in tqdm(batches, desc=f"Processing batches in {base_path}", total=len(batches), leave=True, dynamic_ncols=True):
                if batch.is_dir():
                    checker = self.checker_class(batch)
                    batch_info = checker.check_batch()
                    if batch_info:
                        all_batches_details.append(batch_info)
        return all_batches_details

    def create_dataframe(self, batch_details: List[Dict[str, any]]) -> pd.DataFrame:
        """
        Create a DataFrame from the batch details.
        """
        return pd.DataFrame(batch_details)

    def save_report(self, cfg: DictConfig, batch_details: List[Dict[str, any]], filename_prefix: str) -> None:
            """
            Save the report to a CSV file with a timestamp.
            """
            df = self.create_dataframe(batch_details)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_save_dir = Path(cfg.paths.data_dir, "storage_lockers")
            csv_save_dir.mkdir(exist_ok=True, parents=True)
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_save_dir / csv_filename, index=False)
            log.info(f"Report saved to {csv_filename}")

def generate_and_save_report(cfg: DictConfig, paths: List[Path], checker_class: Type[BaseBatchChecker], filename_prefix: str) -> None:
    """
    Generate and save the report for the given paths and checker class.
    """
    report = BatchReport(paths, checker_class)
    batch_details = report.generate_report()
    report.save_report(cfg, batch_details, filename_prefix)

def main(cfg: DictConfig) -> None:
    """
    Main function to execute the BlobMetricExporter.
    """
    log.info(f"Starting {cfg.task}")

    # Define paths for batches with subdirectories
    paths_with_subdirectories = [
        Path(cfg.paths.longterm_images, "semifield-developed-images"),
        Path(cfg.paths.GROW_DATA, "semifield-developed-images"),
    ]

    # Define paths for batches without subdirectories
    paths_without_subdirectories = [
        Path(cfg.paths.longterm_images, "semifield-cutouts"),
        Path(cfg.paths.GROW_DATA, "semifield-cutouts"),
    ]

    # Generate and save report for batches with subdirectories
    generate_and_save_report(cfg, paths_with_subdirectories, SemifieldDevelopedBatchChecker, "semif_developed_batch_details")

    # Generate and save report for batches without subdirectories
    generate_and_save_report(cfg, paths_without_subdirectories, SemifieldCutoutBatchChecker, "semif_cutouts_batch_details")