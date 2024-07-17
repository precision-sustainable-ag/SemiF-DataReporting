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
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BlobMetricExporter with configuration data.
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
                cmd = f'azcopy ls "{url + sas}" | cut -d/ -f 1-{n} | awk \'!a[$0]++\' > {outputtxt}'
                os.system(cmd)

    def organize_blob_list(self) -> None:
        """Organizes the list of blobs. (Implementation pending)"""
        pass

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    log.info(f"Starting {cfg.task}")
    exporter = ExporterBlobMetrics(cfg)
    exporter.run_azcopy_ls()
    log.info(f"{cfg.task} completed.")
