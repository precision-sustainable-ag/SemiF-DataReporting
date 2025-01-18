import re
import yaml
import logging
from tqdm import tqdm
from datetime import datetime

log = logging.getLogger(__name__)

def read_yaml(path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")

class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def format_az_file_list(main_file, unprocessed_folder_list=None, processed_folder_list=None):
    """
    args: 
    filename: text file created by ExportBlobMetrics.run_azcopy_ls
    returns:
    data in the following format
    {
        '<batch_prefix>': {
            '<batch_name>': {
                'files': [(<filename>,<filesize>), (<filename>,<filesize>)],
                'processed': <True/False>,
            }
        }
    }
    """
    output = {}
    size_conversion = {"B": 1/1024**3, "KiB": 1/1024**2, "MiB": 1/1024, "GiB": 1}

    filelines =  open(main_file, 'r').readlines()
    for line in filelines:
        if not line.strip():
            continue
        
        # ignores 'INFO: ' and 'azcopy' from other azcopy versions
        line = line.replace("INFO: ", "")
        if "azcopy" in line:
            continue
        filename, size_string = line.replace('\n','').split(';')
        size_regex = r'Content Length: ([\d.]+) (\w+)'
        match = re.search(size_regex, size_string)
        if not match:
            log.warn(f'No size info found: {filename}')
        num,unit = match.groups()
        filesize = float(num) * size_conversion[unit]
        # print(filename, filesize)
        path_splits = filename.split('/')
        # if (unprocessed_folder_list and any(part in unprocessed_folder_list for part in path_splits)) or not unprocessed_folder_list:
        # should only look at these files
        if not unprocessed_folder_list:
            # try catch block to handle and ignore cases like: MD-2024-04-11
            try:
                # ignore these batches instead (if _2 exists)
                path_split = path_splits[0].split('_')
                if len(path_split) == 2 and bool(datetime.strptime(path_split[1], "%Y-%m-%d")):
                    batch_loc, batch_date = path_split # [:2] added to handle cases like TX_2023-09-11_2
                    if not batch_loc in output:
                        output[batch_loc] = {
                            f"{batch_loc}_{batch_date}": {
                                'files': [(filename, filesize)],
                                'has_processed_folders': False if not unprocessed_folder_list else True
                            }
                        }
                    elif f"{batch_loc}_{batch_date}" in output[batch_loc]:
                        output[batch_loc][f"{batch_loc}_{batch_date}"]['files'].append((filename,filesize))
                        # if not output[batch_loc][batch_date]['processed'] and any(part in processed_data_folders for part in filename.split('/')):
                        if not output[batch_loc][f"{batch_loc}_{batch_date}"]['has_processed_folders'] and unprocessed_folder_list:
                            output[batch_loc][f"{batch_loc}_{batch_date}"]['has_processed_folders'] = True
                    else:
                        # batch_loc is present but batch_date is not
                        output[batch_loc][f"{batch_loc}_{batch_date}"] = {
                            'files':[(filename,filesize)],
                            'has_processed_folders': False if not unprocessed_folder_list else True
                        }
            # except Exception as e:
            #     log.warn(f"Couldn't process file: {filename}")
            finally:
                continue
        elif any(part in unprocessed_folder_list for part in path_splits):
            try:
                path_split = path_splits[0].split('_')
                if len(path_split) == 2 and bool(datetime.strptime(path_split[1], "%Y-%m-%d")):
                    batch_loc, batch_date = path_split # [:2] added to handle cases like TX_2023-09-11_2
                    if not batch_loc in output:
                        output[batch_loc] = {
                            f"{batch_loc}_{batch_date}": {
                                'files': [(filename, filesize)],
                                'has_processed_folders': True if any(part in processed_folder_list for part in path_splits) else False
                            }
                        }
                    elif f"{batch_loc}_{batch_date}" in output[batch_loc]:
                        output[batch_loc][f"{batch_loc}_{batch_date}"]['files'].append((filename,filesize))
                        if not output[batch_loc][f"{batch_loc}_{batch_date}"]['has_processed_folders'] and any(part in processed_folder_list for part in path_splits):
                            output[batch_loc][f"{batch_loc}_{batch_date}"]['has_processed_folders'] = True
                    else:
                        # batch_loc is present but batch_date is not
                        output[batch_loc][f"{batch_loc}_{batch_date}"] = {
                            'files':[(filename,filesize)],
                            'has_processed_folders': True if any(part in processed_folder_list for part in path_splits) else False
                        }
            # except Exception as e:
            #     log.warn(f"Couldn't process file: {filename}")
            finally:
                continue
    
    return output


def az_get_batches_size(data, batch_names):
    total_size = 0
    batch_details = [set(batch_name.split('_')) for batch_name in batch_names]
    for batch_prefix in data:
        for batch_name, batch_info in data[batch_prefix].items():
            if batch_name in batch_names:
                    total_size += batch_info['total_size']
    return total_size