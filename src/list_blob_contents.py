import logging
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import os
from utils.utils import read_yaml, format_az_file_list, az_get_batches_size
from concurrent.futures import ProcessPoolExecutor
import subprocess

log = logging.getLogger(__name__)


class ExporterBlobMetrics:
    def __init__(self, cfg: DictConfig) -> None:
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.output_dir = Path(cfg.paths.data_dir, "blob_containers")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.azcopy_cmd = "./azcopy" if os.path.exists("./azcopy") else "azcopy"
        self.task_config = cfg.list_blob_contents

    def run_command(self, command: str) -> str:
        try:
            result = subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            log.error(f"Command failed: {e}")
            return ""

    def process_blob_with_python(self, k: str, blob_data: dict) -> tuple:
        """
        Process a single blob, returning the key and its processed output.
        """
        if k == "account_url":
            return k, ""  # Skip processing for the account URL
        url = blob_data["url"]
        sas = blob_data["sas_token"]
        cmd = f'{self.azcopy_cmd} ls "{url + sas}"'
        output = self.run_command(cmd)

        # Process results in Python
        processed_output = set()
        for line in output.splitlines():
            parts = line.split("/")  # Modify slicing [:2] if needed
            processed_output.add("/".join(parts))

        return k, "\n".join(processed_output)

    def run_azcopy_ls(self) -> None:
        results = {}
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_blob_with_python, k, blob_data)
                for k, blob_data in self.__auth_config_data["blobs"].items()
            ]
            for future in tqdm(futures, desc="Processing blobs"):
                k, processed_output = future.result()
                if processed_output:  # Only add non-empty results
                    results[k] = processed_output

        # Write all results to disk at once
        for k, output in results.items():
            outputtxt = Path(self.output_dir, k + ".txt")
            with open(outputtxt, "w") as f:
                f.write(output)
                
    def get_data_splits(self):
        """Retrieve and return lists of cutout and developed images."""
        # cutouts_blob_data = format_az_file_list(os.path.join(self.output_dir, 'semifield-cutouts.txt'))
        # read az data for semif-uploads
        uploads_blob_data = format_az_file_list(os.path.join(self.output_dir,'semifield-uploads.txt'))
        developed_blob_data = format_az_file_list(
            os.path.join(self.output_dir, 'semifield-developed-images.txt'), 
            self.task_config['unprocessed_folders'], 
            self.task_config['processed_folders'])

        # uploads vs developed -> preprocessed
        
        # if batch in both places (semif-uploads + semif-developed) - preprocessed/processed (if processed folders exist)
        # else unpreprocessed
        
        uploads_blob_data = {k: v for k, v in uploads_blob_data.items() if k in self.task_config['batch_prefixes']}  # cleanup
        preprocessed_batches, unpreprocessed_batches = [], []
        for batch_prefix, batches in uploads_blob_data.items():
            # calculate total size for semif-uploads folder
            # check if files have jpg extension (can be expanded + config needs rework)
            for batch_name, batch_info in batches.items():
                total_size = sum(size for _, size in batch_info['files'])
                batch_info['total_size'] = total_size
                if any(os.path.splitext(file)[1].lower() in set(self.task_config['file_extensions']['images']) for file,_ in batch_info['files']):
                    preprocessed_batches.append(f"{batch_prefix}_{batch_name}")
                else:
                    unpreprocessed_batches.append(f"{batch_prefix}_{batch_name}")
        
        unpreprocessed_size = az_get_batches_size(uploads_blob_data, unpreprocessed_batches)
        preprocessed_size = az_get_batches_size(uploads_blob_data, preprocessed_batches)

        log.info(f"Found {len(unpreprocessed_batches)} un-preprocessed batches with total size of {unpreprocessed_size/1024} GiB")
        log.info(f"Found {len(preprocessed_batches)} preprocessed batches with total size of {preprocessed_size/1024} GiB")
        log.info(f"{unpreprocessed_batches}")

        developed_blob_data = {k: v for k, v in developed_blob_data.items() if k in self.task_config['batch_prefixes']}  # cleanup
        unprocessed_batches, processed_batches = [], []
        for batch_prefix, batches in developed_blob_data.items():
            for batch_name, batch_info in batches.items():
                total_size = sum(size for _, size in batch_info['files'])
                batch_info['total_size'] = total_size
                if batch_info['processed']:
                    processed_batches.append(f"{batch_prefix}_{batch_name}")
                else:
                    unprocessed_batches.append(f"{batch_prefix}_{batch_name}")

        unprocessed_size = az_get_batches_size(uploads_blob_data, unprocessed_batches)
        processed_size = az_get_batches_size(uploads_blob_data, processed_batches)

        log.info(f"Found {len(unprocessed_batches)} un-processed batches with total size of {unprocessed_size/1024} GiB")
        log.info(f"Found {len(processed_batches)} processed batches with total size of {processed_size/1024} GiB")
        log.info(f"unpre: {unpreprocessed_batches}, unpro: {unprocessed_batches}")
        log.info(f"pre but not pro: {len(set(preprocessed_batches) - set(processed_batches))}")

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    exporter = ExporterBlobMetrics(cfg)
    # exporter.run_azcopy_ls()
    log.info("Extracting data completed.")
    exporter.get_data_splits()
