import re
import yaml
import logging
from tqdm import tqdm

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

def format_az_file_list(main_file):
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
                'total_size': <size in MiB>
            }
        }
    }
    """
    # TODO: should the size be in GB/MB?
    # TODO: processed_data_folders needs to be verified
    processed_data_folders = {'autosfm', 'metadata', 'masks'}
    output = {}
    size_conversion = {"B": 1/1024/2024, "KiB": 1/1024, "MiB": 1, "GiB": 1024}

    filelines =  open(main_file, 'r').readlines()
    for line in filelines:
        filename, size_string = line.replace('\n','').split(';')
        size_regex = r'Content Length: ([\d.]+) (\w+)'
        match = re.search(size_regex, size_string)
        if not match:
            log.warn(f'No size info found: {filename}')
        num,unit = match.groups()
        filesize = float(num) * size_conversion[unit]
        # print(filename, filesize)
        path_splits = filename.split('/')
        # try catch block to handle and ignore cases like: MD-2024-04-11
        try:
            batch_loc, batch_date = path_splits[0].split('_')[:2] # [:2] added to handle cases like TX_2023-09-11_2
            if not batch_loc in output:
                output[batch_loc] = {
                    batch_date: {
                        'files': [(filename, filesize)],
                        'processed': True if any(part in processed_data_folders for part in filename.split('/')) else False
                    }
                }
            elif batch_date in output[batch_loc]:
                output[batch_loc][batch_date]['files'].append((filename,filesize))
                if not output[batch_loc][batch_date]['processed'] and any(part in processed_data_folders for part in filename.split('/')):
                    output[batch_loc][batch_date]['processed'] = True
            else:
                # batch_loc is present but batch_date is not
                output[batch_loc][batch_date] = {
                    'files':[(filename,filesize)],
                    'processed': True if any(part in processed_data_folders for part in filename.split('/')) else False
                }
        except Exception as e:
            log.warn(f"Couldn't process file: {filename}")
        finally:
            continue
    
    # add total size per batch to the output
    for _, batches in output.items():
        for _, batch_info in batches.items():
            total_size = sum(size for _, size in batch_info['files'])
            batch_info['total_size'] = total_size
    return output

