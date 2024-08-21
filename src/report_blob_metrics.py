import pandas as pd
import logging
from fpdf import FPDF
from omegaconf import DictConfig
from pathlib import Path
import csv
import os

log = logging.getLogger(__name__)

class ReporterBlobMetrics:
    """
    Reports blob metrics to PDF.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BlobMetricReporter with configuration data.
        args:
            cfg: DictConfig - The configuration data.

        Arguments:
        report_dir: The directory where the report will be saved.
        output_dir: The directory where the output data is saved.
        blob_containers: A boolean indicating whether the data is from blob containers.
        pdf: The FPDF object used to generate the PDF report.
        """
        self.report_dir = cfg.paths.report
        self.output_dir = cfg.paths.data_dir
        self.blob_containers = 1#cfg.report.blob_containers
        self.pdf = FPDF()
        log.info("Initialized ReporterBlobMetrics with configuration data.")

    def generate_pdf_report(self, result_df_list: dict[pd.DataFrame], output_path: Path) -> None:
        """ Generates a PDF report with the given data and saves it to the output path.
        args:
            result_df: The data to be included in the report.
            output_path: The path where the report will be saved.
        """

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        ##################### ADD images to the PDF #####################
        self.pdf.cell(200, 10, txt="Average Crop Counts for Each Season", ln=True, align='C')
        image=Path(self.output_dir, 'average_season_count.png')
        self.pdf.image(image, x=5, y=25, w=190, h=125)

        self.pdf.add_page()
        self.pdf.cell(200, 10, txt="Average Crop Counts for Each Month", ln=True, align='C')
        image=Path(self.output_dir, 'average_month_count.png')
        self.pdf.image(image, x=5, y=25, w=190, h=125)

        self.pdf.add_page()
        self.pdf.cell(200, 10, txt="Average Batch Counts for Each Month", ln=True, align='C')
        image=Path(self.output_dir, 'average_batch_count.png')
        self.pdf.image(image, x=5, y=25, w=190, h=125)

        self.pdf.add_page()
        self.pdf.cell(200, 10, txt="Average Cut out Counts for Each Month", ln=True, align='C')
        image=Path(self.output_dir, 'average_cut_out_count.png')
        self.pdf.image(image, x=5, y=25, w=190, h=125)

        self.pdf.add_page()
        self.pdf.cell(200, 10, txt="Total ARW Counts for Each Month", ln=True, align='C')
        image=Path(self.output_dir, 'dataset_statistics_upload.png')
        self.pdf.image(image, x=5, y=25, w=190, h=125)
        ##################### ADD lists to the PDF #####################
        self.pdf.add_page()
        #adding uncolorized batch names
        self.pdf.cell(200, 10, txt="Uncolorized batches", ln=True, align='C')
        for i in set(result_df_list['uncolorized_batches']['State'].values):
            self.pdf.cell(200, 10, txt=str(i), ln=True, align='C')
            st_list =  result_df_list['uncolorized_batches'][result_df_list['uncolorized_batches']['State']==i]['Batch'].to_list()
            self.pdf.cell(100, 10, txt=str(st_list)[1:-1], ln=True, align='C')

        ##################### ADD tables to the PDF #####################
        line_height = self.pdf.font_size
        self.pdf.set_font("Times", size=10)
        for df_name in result_df_list:

            self.pdf.add_page()
            
            df = result_df_list[df_name]

            col_width = self.pdf.epw /(len(df.columns)+3)
            
            self.pdf.cell(200, 10, txt=df_name, ln=True, align='C') 
            
            #one liner for removing lengthy lists in the last cell to first three files.
            df = df.map(lambda y: y[:3]+['...',] if isinstance(y, list) and len(y)>3 else y )
            
            logging.info(f"Creating the batch statistics report.")
            #create your fpdf table ..

            #add the column names
            self.pdf_row(df.columns, col_width, line_height)
            self.pdf.ln(line_height)
            for j,row in df.iterrows():            
                #choose right height for current row
                if 'Missing' in row and len(row['Missing'])>4:
                    line_height=self.pdf.font_size*3
                else:
                    line_height=self.pdf.font_size
                #draw each cell in the row

                self.pdf_row(row, col_width, line_height)
   
                self.pdf.ln(line_height)
            

        self.pdf.output(output_path)
        log.info(f"PDF report generated and saved to {output_path}.")
    
    def pdf_row(self, row: pd.Series, col_width: float, line_height: float) -> None:
        for i,datum in enumerate(row):   
            #for larger size text (missing file names)                 
            if i== 8:
                self.pdf.multi_cell(col_width*3, line_height, str(datum), border=1,align='L',ln=3, 
            max_line_height=self.pdf.font_size)
            #for batch names
            elif i==0:
                self.pdf.multi_cell(col_width*2, line_height, str(datum), border=1,align='L',ln=3, max_line_height=self.pdf.font_size)
            #for numbers
            else:
                self.pdf.multi_cell(col_width, line_height, str(datum), border=1,align='L',ln=3, max_line_height=self.pdf.font_size)
    
    def save_pdf(self) -> None:
        """Conect csv to a PDF report.
        """
        save_csv_dir = Path(self.output_dir, 'blob_containers')
        all_stats={}
        if self.blob_containers:
            for container in os.scandir(save_csv_dir):
                try:
                    # Read the CSV file
                    mismatch_statistics = pd.read_csv(Path(save_csv_dir, container.name))
                    all_stats[container.name.split('.')[0]]=mismatch_statistics
                except Exception as e:
                    log.error(f"Error reading CSV file: {container.name}")
                    continue
        
        # result_df = reporter.combine_data(image_counts, average_image_counts)
        output_path = Path(self.report_dir,'semifield_report.pdf')
        self.generate_pdf_report(all_stats, output_path)


def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricReporter."""
    log.info(f"Starting {cfg.task}")
    reporter = ReporterBlobMetrics(cfg)
    reporter.save_pdf()

    log.info(f"{cfg.task} completed.")

