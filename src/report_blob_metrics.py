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
                #hardcord to remove MD_2022-06-22
                if batch=='MD_2022-06-22' and file_type in ['json',]and len(full_filename.split('.'))>2:
                    continue
                    # rearrange=full_filename.split('.')[-3].split('_')
                    # file_name='MD_Row-'+rearrange[1]+'_'+rearrange[-1]
                # elif batch=='MD_2022-06-22' and file_type in ['jpg']:
                #     rearrange=full_filename.split('.')[-3].split('_')
                #     file_name='MD_Row-'+rearrange[1]+'_'+rearrange[-1]
                else:
                    file_name=full_filename.split('.')[-2]
            else:
                file_type = 'folder'
                file_name=full_filename

            batches.append((batch, folder_name, file_name, file_type))
        log.info(f"Extracted {len(batches)} batches.")
        return batches
    
    def name_diferences(self,file_name: str, folder_name: str) -> str:
        #rearranges the file name to match the image name
        w=1
        return None


    
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
        return df_copy

    def combine_data(self, image_counts: pd.DataFrame, average_image_counts: pd.DataFrame) -> pd.DataFrame:
        # Combines image counts and average image counts into a single DataFrame.
        result_df = pd.merge(image_counts, average_image_counts, on=['State', 'Month'])
        return result_df

    def generate_pdf_report(self, result_df: pd.DataFrame, output_path: Path) -> None:
        
        # Combines image counts and average image counts into a single DataFrame.
        pdf = FPDF()
        # pdf.add_page()
        # pdf.set_font("Arial", size=12)
        # pdf.cell(200, 10, txt="Image Counts and Averages Report", ln=True, align='C')
        # for index, row in result_df.iterrows():
        #     pdf.cell(200, 5, txt=f"{row['Batch']}", ln=True)
        #     pdf.cell(200, 5, txt=f"Image Count: {row['ImageCount']}", ln=True)
        #     pdf.cell(200, 5, txt=f"Average Image Count: {row['AverageImageCount']:.2f}", ln=True)
        #     pdf.ln(10)

        pdf.add_page()
        pdf.set_font("Times", size=10)
        line_height = pdf.font_size
        col_width = pdf.epw /9

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
            if len(row[5])>1:
                line_height=pdf.font_size*(len(row[5]))
            else:
                line_height=pdf.font_size
            #draw each cell in the row
            for i,datum in enumerate(row):                    
                if i== 5:
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
    mismatch_statistics,unprocessed_batches=calculate_blob_metrics.compute_matching(df_filtered,cfg.matching_folders)
    #writing the mismatch statistics to a csv file for now

    fields=['Batch','images','metadata','meta_mask','isMatching','Missing']
    frames = []
    with open('mismatch_statistics_record.csv', 'w') as csvfile:
        w = csv.DictWriter( csvfile, fields )
        for key,val in sorted(mismatch_statistics.items()):
            row = {'Batch': key}
            row.update(val)
            
            frames.append(pd.DataFrame([row]))
            w.writerow(row)
    
    # result_df = reporter.combine_data(image_counts, average_image_counts)
    output_path = Path(reporter.report_dir,'semifield-developed-images_image_counts_and_averages_report.pdf')
    
    pdf_df=pd.concat(frames).reset_index(drop=True)
    pdf_df.columns=fields
    reporter.generate_pdf_report(pdf_df, output_path)

    log.info(f"{cfg.task} completed.")

