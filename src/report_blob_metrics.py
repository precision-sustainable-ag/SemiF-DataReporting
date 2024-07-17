import pandas as pd
import logging
from fpdf import FPDF
from omegaconf import DictConfig
from pathlib import Path

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
            if line.startswith('INFO: azcopy:'):
                continue
            if line.startswith('INFO: '):
                parts = line.split('/')
                batch = parts[0].replace('INFO: ', '').strip()
                filename = parts[-1].split(";")[0].strip()
                if '.' in filename:
                    file_type = filename.split('.')[-1]
                else:
                    file_type = 'folder'
                batches.append((batch, file_type))
        log.info(f"Extracted {len(batches)} batches.")
        return batches
    
    def remove_invalid_batches(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # Removes invalid batches from DataFrame.
        invalid_pattern = r'^[A-Z]{2}-\d{4}-\d{2}-\d{2}$'
        remaing_rows = df[df[column_name].str.contains(invalid_pattern, regex=True)]
        print(remaing_rows.Batch.unique())
        filtered_df = df[~df[column_name].str.contains(invalid_pattern, regex=True)]
        log.info(f"Filtered out invalid batches. Remaining batches: {len(filtered_df)}.")
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

    def calculate_image_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculates the image counts grouped by state, month, and batch.
        df = self.remove_invalid_batches(df, "Batch")
        # To avoid setting values on a copy of a slice from a DataFrame.
        df_copy = df.copy()
        # Apply the function to extract state and month
        df_copy['State'] = df['Batch'].apply(self.extract_state)
        df_copy['Month'] = df['Batch'].apply(self.extract_month)

        image_counts = df_copy[df_copy['FileType'].isin(['jpg', 'png'])].groupby(['State', 'Month', 'Batch']).size().reset_index(name='ImageCount')
        log.info(f"Calculated image counts for {len(image_counts)} batches.")
        return image_counts

    def calculate_average_image_counts(self, image_counts: pd.DataFrame) -> pd.DataFrame:
        # Calculates the average image counts grouped by state and month.
        average_image_counts = image_counts.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')
        log.info(f"Calculated average image counts for {len(average_image_counts)} state-month groups.")
        return average_image_counts

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
    
    df = pd.DataFrame(batches, columns=['Batch', 'FileType'])
    log.info(f"Created DataFrame with {len(df)} rows.")

    image_counts = reporter.calculate_image_counts(df)
    
    average_image_counts = reporter.calculate_average_image_counts(image_counts)
    result_df = reporter.combine_data(image_counts, average_image_counts)
    output_path = Path(reporter.report_dir,'semifield-developed-images_image_counts_and_averages_report.pdf')
    reporter.generate_pdf_report(result_df, output_path)

    log.info(f"{cfg.task} completed.")

