import pandas as pd
import logging
from fpdf import FPDF
from omegaconf import DictConfig
from pathlib import Path
import calculate_blob_metrics

log = logging.getLogger(__name__)

class ReporterBlobMetrics:
    """
    Reports blob metrics to PDF.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BlobMetricReporter with configuration data.
        """
        self.report_dir = cfg.paths.report
        self.output_dir = cfg.paths.data_dir
        log.info("Initialized ReporterBlobMetrics with configuration data.")
    # Function to extract batch names and file types
    def extract_batches(self, lines: list[str]) -> list[tuple[str, str]]:
        batches = []
        for line in lines:
            # accounts for blank lines
            if not line.strip():
                continue
            # in some versions of azcopy
            if line.startswith('INFO: azcopy:'):
                continue
            
            parts = line.split('/')
            if line.startswith('INFO: '):
                
                batch = parts[0].replace('INFO: ', '').strip()
            else:
                batch = parts[0]
            full_filename = parts[-1].split(";")[0].strip()
            folder_name=parts[-2]
            
            if '.' in full_filename:
                file_type = full_filename.split('.')[-1]
                file_name=full_filename.split('.')[-2]
            else:
                file_type = 'folder'
                file_name=full_filename

            batches.append((batch, folder_name, file_name, file_type))
        log.info(f"Extracted {len(batches)} batches.")
        return batches
    
    def remove_invalid_batches(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # Removes invalid batches from DataFrame.
        valid_pattern = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'
        invalid_batches = df[~df[column_name].str.contains(valid_pattern, regex=True)][column_name].unique()        
        filtered_df = df[df[column_name].str.contains(valid_pattern, regex=True)]
        
        log.info(f"Removed {len(invalid_batches)} unique batches due to invalid pattern.")
        return filtered_df

    def extract_month(self, batch_name: str) -> str:
        # Extracts the month from the batch name.
        parts = batch_name.split('_')
        month = parts[1][:7]
        return month

    def extract_state(self, batch_name: str) -> str:
        # Extracts the state from the batch name.
        state = batch_name.split('_')[0]
        return state
    
        # Function to format and filter data
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculates the image counts grouped by state, month, and batch.
        df = self.remove_invalid_batches(df, "Batch")
        # To avoid setting values on a copy of a slice from a DataFrame.
        df_copy = df.copy()
        # Apply the function to extract state and month
        df_copy['State'] = df['Batch'].apply(self.extract_state)
        df_copy['Month'] = df['Batch'].apply(self.extract_month)

        print(df_copy)
        return df_copy

    def combine_data(self, image_counts: pd.DataFrame, average_image_counts: pd.DataFrame) -> pd.DataFrame:
        # Combines image counts and average image counts into a single DataFrame.
        result_df = pd.merge(image_counts, average_image_counts, on=['State', 'Month'])
        return result_df

    def generate_pdf_report(self, result_df: pd.DataFrame, output_path: Path) -> None:
        # Combines image counts and average image counts into a single DataFrame.
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Image Counts and Averages Report", ln=True, align='C')
        for index, row in result_df.iterrows():
            pdf.cell(200, 5, txt=f"{row['Batch']}", ln=True)
            pdf.cell(200, 5, txt=f"Image Count: {row['ImageCount']}", ln=True)
            pdf.cell(200, 5, txt=f"Average Image Count: {row['AverageImageCount']:.2f}", ln=True)
            pdf.ln(10)
        pdf.output(output_path)
        log.info(f"PDF report generated and saved to {output_path}.")

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    log.info(f"Starting {cfg.task}")
    reporter = ReporterBlobMetrics(cfg)

    # Load the text file
    file_path = Path(cfg.paths.data_dir,'semifield-developed-images.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Main execution
    batches = reporter.extract_batches(lines)
    df = pd.DataFrame(batches, columns=['Batch', 'FolderName', 'FileName', 'FileType'])
    log.info(f"Created DataFrame with {len(df)} rows.")

    df_filtered = reporter.format_data(df)

    # Calculate image counts and averages
    image_counts = calculate_blob_metrics.calculate_image_counts(df_filtered)
    average_image_counts = calculate_blob_metrics.calculate_average_image_counts(image_counts)

    #Compare file type lengths
    # calculate_blob_metrics.compute_matching(df_filtered)

    result_df = reporter.combine_data(image_counts, average_image_counts)
    output_path = Path(reporter.report_dir,'semifield-developed-images_image_counts_and_averages_report.pdf')
    reporter.generate_pdf_report(result_df, output_path)

    log.info(f"{cfg.task} completed.")

