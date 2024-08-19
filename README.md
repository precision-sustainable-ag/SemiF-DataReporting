# SemiF-DataReporting
This repository provides detailed reports on Semifield image data, including data contents, species distribution, temporal and spatial distribution, missing data analysis, and status of unprocessed or backlog data.

## Installation and Setup

### Installing Conda
To manage the project's dependencies efficiently, we use Conda, a powerful package manager and environment manager. Follow these steps to install Conda if you haven't already:

1. Download the appropriate version of Miniconda for your operating system from the official [Miniconda website](https://docs.anaconda.com/free/miniconda/).
2. Follow the installation instructions provided on the website for your OS. This typically involves running the installer from the command line and following the on-screen prompts.
3. Once installed, open a new terminal window and type `conda list` to ensure Conda was installed correctly. You should see a list of installed packages.


### Setting Up Your Environment Using an Environment File
After installing Conda, you can set up an environment for this project using an environment file, which specifies all necessary dependencies. Here's how:

1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Locate the `environment.yaml` file in the repository. This file contains the list of packages needed for the project.
4. Create a new Conda environment by running the following command:
   ```bash
   conda env create -f environment.yaml
   ```
   This command reads the `environment.yaml` file and creates an environment with the name and dependencies specified within it.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.


### Running the Script
With the environment set up and activated, you can run the scripts provided in the repository to begin data exploration and analysis:

1. Ensure your Conda environment is activated:
   ```
   conda activate semifield-reports
   ```
2. [NOTE] Setup the pipeline in the main [config](conf/config.yaml#L11). To run a script, use the following command syntax:
   ```bash
   python main.py task=<task_name>
   ```


## Major Scripts

### `export_blob_metrics.py`

1. ExporterBlobMetrics: Exports blob metrics by running AzCopy commands and saving the output to text files.

2. CalculatorBlobMetrics: Calculates and analyzes blob metrics from the exported text files, including extracting batch details, filtering data, and computing image counts.

### Usage

1. Run the script with the configuration file as an argument:

    ```bash
    python main.py task=export_blob_metrics
    ```

### Output

1. Text Files: The ExporterBlobMetrics class saves blob lists as text files. The text files are saved in the directory specified by `cfg.paths.data_dir` in the configuration file, with the nameing format `<blob_container_name>.txt`.

2. CSV Report: The CalculatorBlobMetrics class generates a CSV file containing mismatch statistics, detailing any discrepancies found during analysis. The CSV files are saved in the directory specified by `cfg.paths.data_dir` in the configuration file.
 
### `plot_blob_metrics.py`

1. PlotBlobMetrics : This class is designed to visualize blob metrics by generating various plots such as bar charts and line charts, using data loaded from CSV files.

### Usage

1. Run the script with the configuration file as an argument:

    ```bash
    python main.py task=plot_blob_metrics
    ```

### Output

1. Plots: The script generates and saves various plots for reporting. The plots are saved as PNG files in the directory specified by `cfg.paths.data_dir`.

### `report_blob_metrics.py`

1. ReporterBlobMetrics: Generates PDF reports from blob metrics stored in CSV files and saves the reports.

### Usage

1. Run the script with the configuration file as an argument:

    ```bash
    python main.py task=report_blob_metrics
    ```

### Output

1. PDF Report: The ReporterBlobMetrics class generates a PDF report containing the blob metrics. The PDF file is saved in the directory specified by `cfg.paths.report` in the configuration file, with the naming format `semifield_report.pdf`.