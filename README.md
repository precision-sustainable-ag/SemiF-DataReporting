# SemiF-DataReporting
This repository provides detailed reports on Semifield image data, including data contents, species distribution, temporal and spatial distribution, missing data analysis, and status of unprocessed or backlog data.


## Major Scripts

### `list_blob_contents.py`

This script processes blob storage containers by leveraging the azcopy tool to list contents and organize them for further analysis. It reads configuration details, including authentication keys and storage paths, to access blob URLs and SAS tokens. Using a ProcessPoolExecutor for parallelism, it processes each blob's output, structures the data, and saves it as text files in an organized output directory. The script ensures efficient handling of blob data and logs progress to assist with monitoring and troubleshooting.
