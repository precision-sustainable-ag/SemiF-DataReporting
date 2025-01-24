import pandas as pd
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
import re
import os
from typing import List, Dict, Optional, Tuple, Type
from datetime import datetime
from tqdm import tqdm
from utils.utils import TqdmLoggingHandler, _get_bbot_version
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
    
    def _get_dir_size(self, dir_path: Path) -> int:
        """
        Recursively calculate the total size of a directory using os.scandir.
        Return the size in bytes.
        """
        total_size = 0
        if not Path(dir_path).exists():
            return total_size
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file():
                    total_size += entry.stat().st_size
                elif entry.is_dir():
                    total_size += self._get_dir_size(entry.path)
        return total_size

    def get_folder_size(self, subfolder: Path) -> float:
        """
        Calculate the size of a specific subfolder (e.g., images, metadata, meta_masks).
        Return the size in GiB.
        """
        return self._get_dir_size(subfolder) / (1024 ** 3)  # Convert bytes to GiB


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
        log.debug("Getting all files in batch path...")
        all_files = list(self.batch_path.glob('*'))

        log.debug("Processing file extensions...")
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

    def get_file_sizes(self) -> Dict[str, float]:
        """
        Calculate the total size of .jpg, .png (excluding _mask.png), .json, and _mask.png files
        in the batch folder. Return the sizes in GiB.
        """
        file_sizes = {
            "jpg_size": 0,
            "png_size": 0,
            "json_size": 0,
            "mask_size": 0
        }

        # Traverse the directory using os.scandir for efficient file listing
        for entry in os.scandir(self.batch_path):
            if entry.is_file():
                if entry.name.endswith(".jpg"):
                    file_sizes["jpg_size"] += entry.stat().st_size
                elif entry.name.endswith(".json"):
                    file_sizes["json_size"] += entry.stat().st_size
                elif entry.name.endswith(".png"):
                    if "_mask" in entry.name:
                        file_sizes["mask_size"] += entry.stat().st_size
                    else:
                        file_sizes["png_size"] += entry.stat().st_size

        # Convert sizes to GiB
        for key in file_sizes:
            file_sizes[key] = file_sizes[key] / (1024 ** 3)  # Convert bytes to GiB

        return file_sizes

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
        Check the batch for subfolder counts, matching counts, unprocessed status, format type,
        missing files, and add the size of each subfolder (images, metadata, meta_masks) using multithreading.
        """
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.debug(f"Valid batch found: {self.batch_path.name}")
            images_count, metadata_count, meta_masks_count = self.counter.get_subfolder_counts()
            has_matching = self.check_has_matching(images_count, metadata_count, meta_masks_count)
            unprocessed = self.check_unprocessed(self.counter.metadata_path, self.counter.meta_masks_path)
            format_type = self.determine_format_type(self.counter.metadata_path, "annotations")

            # Use ThreadPoolExecutor to get subfolder sizes concurrently
            with ThreadPoolExecutor() as executor:
                futures = {
                    "ImagesFolderSizeGiB": executor.submit(self.counter.get_folder_size, self.counter.images_path),
                    "MetadataFolderSizeGiB": executor.submit(self.counter.get_folder_size, self.counter.metadata_path),
                    "MetaMasksFolderSizeGiB": executor.submit(self.counter.get_folder_size, self.counter.meta_masks_path),
                }
                subfolder_sizes = {key: future.result() for key, future in futures.items()}

            batch_info = {
                "path": str(self.batch_path.parent.parent.name),
                "batch": self.batch_path.name,
                "images": images_count,
                "metadata": metadata_count,
                "meta_masks": meta_masks_count,
                "HasMatching": has_matching,
                "UnProcessed": unprocessed,
                "FormatType": format_type,
                **subfolder_sizes  # Add folder sizes
            }

            return batch_info
        else:
            log.warning(f"Ignoring invalid batch: {self.batch_path.name}")
        return None

class UploadsCounter(BaseCounter):
    def __init__(self, batch_path:Path):
        super().__init__(batch_path)
        self.subfolders = [item for item in batch_path.iterdir() if item.is_dir()]
    
    def get_subdir_size(self):
        total_subfolder_size = 0
        for subfolder in self.subfolders:
            total_subfolder_size += super()._get_dir_size(subfolder)
        return total_subfolder_size
    
    def get_counts(self):
        if len(self.subfolders) == 0:
            jpg_count: int = len(list(self.batch_path.glob('*.jpg'))) + len(list(self.batch_path.glob('*.JPG')))
            png_count: int = len(list(self.batch_path.glob('*.png'))) + len(list(self.batch_path.glob('*.PNG')))
            arw_count: int = len(list(self.batch_path.glob('*.arw'))) + len(list(self.batch_path.glob('*.ARW')))
            raw_count: int = len(list(self.batch_path.glob('*.raw'))) + len(list(self.batch_path.glob('*.RAW')))
        else:
            jpg_count, png_count, arw_count, raw_count = 0,0,0,0
            for subfolder in self.subfolders:
                jpg_count += len(list(subfolder.glob('*.jpg'))) + len(list(subfolder.glob('*.JPG')))
                png_count: int = len(list(subfolder.glob('*.png'))) + len(list(subfolder.glob('*.PNG')))
                arw_count += len(list(subfolder.glob('*.arw'))) + len(list(subfolder.glob('*.ARW')))
                raw_count += len(list(subfolder.glob('*.raw'))) + len(list(subfolder.glob('*.RAW')))
        return jpg_count, png_count, arw_count, raw_count

class SemifieldUploadsBatchChecker(BaseBatchChecker):
    def __init__(self, batch_path: Path, developed_batches_paths: List[Path]):
        super().__init__(batch_path)
        # no subfolders
        self.counter = UploadsCounter(batch_path)
        self.developed_batches_paths = developed_batches_paths

    # there are batches that are present in multiple lts locations
    # currently, there are duplicate records
    # ex: MD_2023-07-21 -- unpreprocessed
    # MD_2023-07-07

    # duplicates -> match the lts location with developed batches folder of that batch
    # don't use png counts
    # config has been updated with new dates for bbot versions
    # 
    def is_preprocessed(self):
        """Check if this batch exists in any of the semifield-developed paths"""
        batch_name = self.batch_path.name
        for developed_path in self.developed_batches_paths:
            if (developed_path / batch_name).exists():
                return True, developed_path.parent.name
        return False, None

    def check_batch(self) -> Optional[Dict[str, any]]:
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.debug(f"Valid batch found: {self.batch_path.name}")
            is_preprocessed, developed_lts_loc = self.is_preprocessed()
            total_size = self.counter.get_folder_size(self.batch_path)
            jpg_count, png_count, arw_count, raw_count = self.counter.get_counts()
            batch_name_splits = self.batch_path.name.split("_")
            batch_info = {
                "path": str(self.batch_path.parent.parent.name),
                "batch": self.batch_path.name,
                "raw_count": arw_count if arw_count else raw_count,
                "jpg_count": jpg_count,
                "developed_lts_loc": developed_lts_loc,
                # "png_count": png_count,
                "totalSizeGiB": total_size,
                "IsPreprocessed": is_preprocessed,
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
        Check the batch for file sizes (.jpg, .png, .json, and _mask.png) using multithreading.
        """
        if self.validator.is_valid_batch_name(self.batch_path.name):
            log.debug(f"Valid batch found: {self.batch_path.name}")
            jpg_count, png_count, json_count, mask_count = self.counter.get_cutout_counts()
            has_matching = self.check_has_matching(jpg_count, png_count, json_count, mask_count)
            format_type = self.determine_format_type(self.batch_path, "validated")
            # Use ProcessPoolExecutor to calculate sizes concurrently
            with ProcessPoolExecutor() as executor:
                future = executor.submit(self.counter.get_file_sizes)
                file_sizes = future.result()
            

            batch_info = {
                "path": str(self.batch_path.parent.parent.name),
                "batch": self.batch_path.name,
                "jpg_count": jpg_count,
                "png_count": png_count,
                "json_count": json_count,
                "mask_count": mask_count,
                "HasMatching": has_matching,
                "FormatType": format_type,
                "jpg_size_gib": file_sizes["jpg_size"],  # Size of all .jpg files in GiB
                "png_size_gib": file_sizes["png_size"],  # Size of all .png files (excluding masks) in GiB
                "json_size_gib": file_sizes["json_size"],  # Size of all .json files in GiB
                "mask_size_gib": file_sizes["mask_size"],  # Size of all _mask.png files in GiB
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
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp = datetime.now().strftime("%Y%m%d")
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
    log.info(f"Starting local_batch_table_generator")

    paths_with_subdirectories, paths_without_subdirectories = [], []
    paths_uploads = []

    for location in cfg.paths.lts_locations:
        paths_with_subdirectories.append(Path(location, 'semifield-developed-images'))
        paths_without_subdirectories.append(Path(location, 'semifield-cutouts'))
        paths_uploads.append(Path(location, 'semifield-upload'))


    def create_uploads_checker(batch_path: Path) -> SemifieldUploadsBatchChecker:
        return SemifieldUploadsBatchChecker(batch_path, paths_with_subdirectories)
    
    report = BatchReport(paths_uploads, create_uploads_checker)
    batch_details = report.generate_report()
    report.save_report(cfg, batch_details, "semif_upload_batch_details")

    # Generate and save report for batches with subdirectories
    generate_and_save_report(cfg, paths_with_subdirectories, SemifieldDevelopedBatchChecker, "semif_developed_batch_details")

    # Generate and save report for batches without subdirectories
    generate_and_save_report(cfg, paths_without_subdirectories, SemifieldCutoutBatchChecker, "semif_cutouts_batch_details")
    
    # generate_and_save_report(cfg, paths_uploads, SemifieldCutoutBatchChecker(paths_with_subdirectories), "semif_uploads_batch_details")