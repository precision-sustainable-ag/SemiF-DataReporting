import logging
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import os
from utils.utils import read_yaml, format_az_file_list
from concurrent.futures import ProcessPoolExecutor
import subprocess

log = logging.getLogger(__name__)


class ExporterBlobMetrics:
    def __init__(self, cfg: DictConfig) -> None:
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.output_dir = Path(cfg.paths.data_dir, "blob_containers")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.azcopy_cmd = "./azcopy" if os.path.exists("./azcopy") else "azcopy"

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
        cutouts_blob_data = format_az_file_list(os.path.join(self.output_dir, 'semifield-cutouts.txt'))
        developed_blob_data = format_az_file_list(os.path.join(self.output_dir, 'semifield-developed-images.txt'))
        log.info(cutouts_blob_data.keys())
        log.info(developed_blob_data.keys())

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    exporter = ExporterBlobMetrics(cfg)
    # exporter.run_azcopy_ls()
    log.info("Extracting data completed.")
    exporter.get_data_splits()