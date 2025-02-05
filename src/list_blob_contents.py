import logging
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import os
import json
from utils.utils import read_yaml, format_az_file_list, az_get_batches_size
from concurrent.futures import ProcessPoolExecutor
import subprocess
import pandas as pd
from datetime import datetime

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
        # TODO: look into using sets instead of lists for batches

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
                batch_info['file_count'] = len(batch_info['files'])
                # assume all as unpreprocessed
                unpreprocessed_batches.append(batch_name)
                # if any(os.path.splitext(file)[1].lower() in set(self.task_config['file_extensions']['images']) for file,_ in batch_info['files']):
                #     preprocessed_batches.append(batch_name)
                # else:
                #     unpreprocessed_batches.append(batch_name)
        
        preprocessed_batches = unpreprocessed_batches

        developed_blob_data = {k: v for k, v in developed_blob_data.items() if k in self.task_config['batch_prefixes']}  # cleanup
        processed_batches = []
        for batch_prefix, batches in developed_blob_data.items():
            for batch_name, batch_info in batches.items():
                total_size = sum(size for _, size in batch_info['files'])
                batch_info['total_size'] = total_size
                batch_info['file_count'] = len(batch_info['files'])
                # assume all as processed
                processed_batches.append(batch_name)
                # if batch_info['processed']:
                #     processed_batches.append(batch_name)
                # else:
                #     unprocessed_batches.append(batch_name)
        
        # set logic:
        # 1. all in develop assumed as processed, if exists, cannot be unpreprocessed
        # 2. preprocessed - was same as unpreprocessed, but unpre updated, so update
        # 3. exists in both - (unpre+pre) intersection (processed) - pro is currently all in developed
        unpreprocessed_batches = list(set(unpreprocessed_batches) - set(processed_batches))
        preprocessed_batches = list(set(preprocessed_batches) - set(unpreprocessed_batches))
        
        unprocessed_batches, processed_batches = [], []
        for batch_prefix, batches in developed_blob_data.items():
            for batch_name, batch_info in batches.items():
                processed_batches.append(batch_name) if batch_info['has_processed_folders'] else unprocessed_batches.append(batch_name)
        
        unpreprocessed_size = az_get_batches_size(uploads_blob_data, unpreprocessed_batches)
        preprocessed_size = az_get_batches_size(uploads_blob_data, preprocessed_batches)
        unprocessed_size = az_get_batches_size(uploads_blob_data, unprocessed_batches)
        processed_size = az_get_batches_size(uploads_blob_data, processed_batches)

        batches_info = [
            ("unpreprocessed", unpreprocessed_batches, unpreprocessed_size),
            ("preprocessed", preprocessed_batches, preprocessed_size),
            ("unprocessed", unprocessed_batches, unprocessed_size),
            ("processed", processed_batches, processed_size)
        ]
        # create a new file for every run and then append to it
        with open(os.path.join(self.output_dir, f'semif-HighLevelStats.txt'),
                  'w') as f:
            f.write("")
        for batch_type, batches, total_size in batches_info:
            log_string = f"Found {len(batches)} {batch_type} batches with total size of {total_size/1024} TiB"
            log.info(log_string)
            with open(os.path.join(self.output_dir, f'semif-HighLevelStats.txt'), 'a') as f:
                f.write(log_string +'\n')
            data = []
            if 'preprocessed' in batch_type:
                for batch_name in batches:
                    batch_prefix, _ = batch_name.split('_')
                    data.append(uploads_blob_data[batch_prefix][batch_name])
            elif 'processed' in batch_type:
                for batch_name in batches:
                    batch_prefix, _ = batch_name.split('_')
                    data.append(developed_blob_data[batch_prefix][batch_name])
            with open(os.path.join(self.output_dir, f'semifield-{batch_type}.jsonl'), 'w') as file:
                file.writelines(json.dumps(item) + '\n' for item in data)

class GenerateProcessedCSV():
    def __init__(self, cfg: DictConfig):
        self.output_dir = Path(cfg.paths.data_dir, "blob_containers")
        self.task_config = cfg.list_blob_contents
        # self.processed_folders = self.task_config.processed_folders
    
    def _file_parts(self, path):
        allparts = []
        while True:
            parts = os.path.split(path)
            if parts[0] == path:  # absolute path
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path:  # relative path
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return [x for x in allparts if x]  # Remove empty strings

    def _get_subfolder_details(self, batches):
        data = []
        for batch_json in batches:
            details = {}
            for subfolder in self.task_config.unprocessed_folders:
                details[subfolder] = {
                    'count': 0,
                    'size': 0
                }
            # print(details)
            batch_name = batch_json['files'][0][0].split('/')[0]
            for file, size in batch_json['files']:
                file_parts = [x.lower() for x in self._file_parts(file)]
                # print(file_parts)
                if file_parts[1] in self.task_config['file_extensions']:
                    if os.path.splitext(file)[1] in set(self.task_config['file_extensions'][file_parts[1]]):
                        details[file_parts[1]]['count'] += 1
                        details[file_parts[1]]['size'] += size
                elif f"{file_parts[1]}/{file_parts[2]}" in self.task_config['file_extensions']:
                    if os.path.splitext(file)[1] in set(self.task_config['file_extensions'][f"{file_parts[1]}/{file_parts[2]}"]):
                        details[file_parts[1]]['count'] += 1
                        details[file_parts[1]]['size'] += size
            data.append({
                'path': 'azure',
                'batch': batch_name,
                'images': details['images']['count'],
                'metadata': details['metadata']['count'],
                'meta_masks': details['meta_masks']['count'],
                'ImagesFolderSizeGiB': details['images']['size'],
                'MetadataFolderSizeGiB': details['metadata']['size'],
                'MetaMasksFolderSizeGiB': details['meta_masks']['size'],
                'UnProcessed': False if batch_json['has_processed_folders'] else True
            })
        return data
    
    def _get_batch_details(self, batch_details):
        batch_name = self._file_parts(batch_details['files'][0][0])[0]
        data = {
            'path': 'azure',
            'batch': batch_name,
            'jpg_count': 0,
            'png_count': 0,
            'json_count': 0,
            'mask_count': 0,
            'jpg_size_gib': 0,
            'png_size_gib': 0,
            'json_size_gib': 0,
            'mask_size_gib': 0
        }
        for filename, filesize in batch_details['files']:
            if os.path.splitext(filename)[1].lower() == '.jpg':
                data['jpg_count'] += 1
                data['jpg_size_gib'] += filesize
            elif os.path.splitext(filename)[1].lower() == '.json':
                data['json_count'] += 1
                data['json_size_gib'] += filesize
            elif filename.lower().endswith('_mask.png'):
                data['mask_count'] += 1
                data['mask_size_gib'] += filesize
            elif os.path.splitext(filename)[1].lower() == '.png':
                data['png_count'] += 1
                data['png_size_gib'] += filesize

        return data
    def _read_semif_cutouts(self):
        # with open(os.path.join(self.output_dir, 'semifield-cutouts.txt'), 'r') as f:
        #     data = [x.replace('\n', '') for x in f.readlines()]
        cutouts_blob_data = format_az_file_list(os.path.join(self.output_dir, 'semifield-cutouts.txt'))
        cutouts_blob_data = {k: v for k, v in cutouts_blob_data.items() if k in self.task_config['batch_prefixes']}
        data = []
        
        # TODO: can be parallelized
        for location, batches in cutouts_blob_data.items():
            for batch_name, details in batches.items():
                data.append(self._get_batch_details(details))
        return data
    
    def _generate_semif_uploads(self, data, is_preprocessed=False):
        output_json = []
        for batch_data in data:
            batch_name = batch_data['files'][0][0].split('/')[0]
            # Count files using list comprehensions
            arw_count = sum(1 for file in batch_data['files'] if file[0].lower().endswith('.arw'))
            raw_count = sum(1 for file in batch_data['files'] if file[0].lower().endswith('.raw'))
            jpg_count = sum(1 for file in batch_data['files'] if file[0].lower().endswith('.jpg'))
            output_json.append({
                "path": 'azure',
                "batch": batch_name,
                "raw_count": arw_count if arw_count else raw_count,
                "jpg_count": jpg_count,
                "totalSizeGiB": batch_data['total_size'],
                "IsPreprocessed": is_preprocessed,
                # "version": self._get_bbot_version(batch_name_splits[0], batch_name_splits[1])
            })
        return output_json

    def create_semif_csv(self):
        with open(os.path.join(self.output_dir, 'semifield-processed.jsonl'), 'r') as f:
            processed_batches = [json.loads(x) for x in f.readlines()]
        with open(os.path.join(self.output_dir, 'semifield-unprocessed.jsonl'), 'r') as f:
            unprocessed_batches = [json.loads(x) for x in f.readlines()]
        
        data = self._get_subfolder_details(processed_batches+unprocessed_batches)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, f'semif_developed_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'))
        log.info("developed images csv written to data location")
        
        data = self._read_semif_cutouts()
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, f'semif_cutouts_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'))
        log.info("cutouts images csv written to data location")

        with open(os.path.join(self.output_dir, 'semifield-preprocessed.jsonl'), 'r') as f:
            preprocessed_batches = [json.loads(x) for x in f.readlines()]
        with open(os.path.join(self.output_dir, 'semifield-unpreprocessed.jsonl'), 'r') as f:
            unpreprocessed_batches = [json.loads(x) for x in f.readlines()]
            
        data = self._generate_semif_uploads(preprocessed_batches, True)
        data.extend(self._generate_semif_uploads(unpreprocessed_batches, False))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, f'semif_upload_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'))
        log.info("uploads csv written to data location")
    
    

        

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    exporter = ExporterBlobMetrics(cfg)
    exporter.run_azcopy_ls()
    log.info("Extracting data completed.")
    # TODO: can this whole thing be removed if the aim is just the csvs
    exporter.get_data_splits()
    log.info("Generated high-level stats and converted data to jsonl format")

    csv_generator = GenerateProcessedCSV(cfg)
    csv_generator.create_semif_csv()
