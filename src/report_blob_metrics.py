import pandas as pd
import logging
from fpdf import FPDF
from omegaconf import DictConfig
from pathlib import Path
import calculate_blob_metrics
import csv

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

    def generate_pdf_report(self, result_df: pd.DataFrame, output_path: Path) -> None:
        """ Generates a PDF report with the given data and saves it to the output path."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # pdf.cell(200, 10, txt="Image Counts and Averages Report", ln=True, align='C')
        # for index, row in result_df.iterrows():
        #     pdf.cell(200, 5, txt=f"{row['Batch']}", ln=True)
        #     pdf.cell(200, 5, txt=f"Image Count: {row['ImageCount']}", ln=True)
        #     pdf.cell(200, 5, txt=f"Average Image Count: {row['AverageImageCount']:.2f}", ln=True)
        #     pdf.ln(10)

        pdf.add_page()
        pdf.set_font("Times", size=10)
        line_height = pdf.font_size
        col_width = pdf.epw /12

        #one liner for removing lengthy lists in the last cell to first three files.
        result_df = result_df.map(lambda y: y[:3]+['...',] if isinstance(y, list) and len(y)>3 else y )
        
        logging.info(f"Creating the batch statistics report.")
        #create your fpdf table ..
        for j,row in result_df.iterrows():
            #add the column names
            if j==0:
                for header in result_df.columns:
                    pdf.cell(col_width, line_height, str(header), border=1)
                pdf.ln(line_height)
            #choose right height for current row
            if len(row['Missing'])>1:
                line_height=pdf.font_size*(len(row['Missing']))
            else:
                line_height=pdf.font_size
            #draw each cell in the row
            for i,datum in enumerate(row):                    
                if i== 8:
                    pdf.multi_cell(col_width*3, line_height, str(datum), border=1,align='L',ln=3, 
                max_line_height=pdf.font_size)
                elif i==0:
                    pdf.multi_cell(col_width*2, line_height, str(datum), border=1,align='L',ln=3, max_line_height=pdf.font_size)
                else:
                    pdf.multi_cell(col_width, line_height, str(datum), border=1,align='L',ln=3, max_line_height=pdf.font_size)
                
            pdf.ln(line_height)

        pdf.output(output_path)
        log.info(f"PDF report generated and saved to {output_path}.")

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricReporter."""
    log.info(f"Starting {cfg.task}")
    reporter = ReporterBlobMetrics(cfg)
    # Read the CSV file
    mismatch_statistics = pd.read_csv(Path(reporter.report_dir, 'mismatch_statistics_record.csv'))

    # result_df = reporter.combine_data(image_counts, average_image_counts)
    output_path = Path(reporter.report_dir,'semifield-developed-images_image_counts_and_averages_report.pdf')
    
    reporter.generate_pdf_report(mismatch_statistics, output_path)

    log.info(f"{cfg.task} completed.")

