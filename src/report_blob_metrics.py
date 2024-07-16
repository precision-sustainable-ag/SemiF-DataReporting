import pandas as pd
import logging
from collections import Counter
from fpdf import FPDF
from omegaconf import DictConfig
from utils.utils import read_yaml
from pathlib import Path
import os

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
    # Function to extract batch names and file types
    def extract_batches(self,lines):
        batches = []
        for line in lines:
            # if line.startswith('INFO: ') and not line.startswith('INFO: azcopy:'):
            #version check
            if line.startswith('INFO: azcopy:'):
                continue
            
            parts = line.split('/')
            if 'INFO:' in parts[0]:
                batch = parts[0].replace('INFO: ', '')
            else:
                batch = parts[0]
            if "Center" in batch:
                continue
            
            filename = parts[-1].split(";")[0]
            if '.' in filename:
                file_type = filename.split('.')[-1]
            else:
                file_type = 'folder'
            
            batches.append((batch, file_type))
        return batches
    def remove_invalid_batches(self,df, column_name):
        invalid_pattern = r'^[A-Z]{2}-\d{4}-\d{2}-\d{2}$'
        filtered_df = df[~df[column_name].str.contains(invalid_pattern, regex=True)]
        return filtered_df

    def extract_month(self,batch_name):
        parts = batch_name.split('_')
        month = parts[1][:7]
        return month

    def extract_state(self,batch_name):
        state = batch_name.split('_')[0]
        return state

    # Function to calculate image counts
    def calculate_image_counts(self,df):
        # pattern = '|'.join(["test_test2", "test_test" , "semi_supervised7", "Center"])
        df = self.remove_invalid_batches(df, "Batch")
        # Apply the function to extract state and month
        df['State'] = df['Batch'].apply(self.extract_state)
        df['Month'] = df['Batch'].apply(self.extract_month)

        print(df)
        # df['State'], df['Month'] = zip(*df['Batch'].apply(lambda x: (x.split('_')[0], x.split('_')[1][:7])))
        image_counts = df[df['FileType'].isin(['jpg', 'png'])].groupby(['State', 'Month', 'Batch']).size().reset_index(name='ImageCount')
        return image_counts

    # Function to calculate average image counts
    def calculate_average_image_counts(self,image_counts):
        average_image_counts = image_counts.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')
        return average_image_counts

    # Function to combine data
    def combine_data(self,image_counts, average_image_counts):
        result_df = pd.merge(image_counts, average_image_counts, on=['State', 'Month'])
        return result_df

    # Function to generate PDF report
    def generate_pdf_report(self,result_df, output_path):
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

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricExporter."""
    log.info(f"Starting {cfg.task}")
    reporter = ReporterBlobMetrics(cfg)
    path_config_data = read_yaml('./conf/paths/default.yaml')
    # Load the text file
    file_path = Path(reporter.output_dir,'semifield-developed-images.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Main execution
    batches = reporter.extract_batches(lines)
    
    df = pd.DataFrame(batches, columns=['Batch', 'FileType'])

    image_counts = reporter.calculate_image_counts(df)
    
    average_image_counts = reporter.calculate_average_image_counts(image_counts)
    result_df = reporter.combine_data(image_counts, average_image_counts)
    output_path = Path(reporter.report_dir,'semifield-developed-images_image_counts_and_averages_report.pdf')
    reporter.generate_pdf_report(result_df, output_path)

    log.info(f"{cfg.task} completed.")

