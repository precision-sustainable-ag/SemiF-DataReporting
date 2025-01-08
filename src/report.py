import logging
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from utils.utils import read_yaml

log = logging.getLogger(__name__)

class Report:
    def __init__(self, cfg: DictConfig) -> None:
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.task_cfg = cfg.report
        self.source_location = cfg.paths.data_dir
        self.report_folder = cfg.paths.report
        self.message_blocks = [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*Data Report* - {datetime.now().strftime('%m/%d/%Y')}"
                }
            },
            {
                'type': 'divider'
            }
        ]
        self.files = []
        
    def copy_relevant_files(self):
        # get relevant blobcontainers files
        # pass in the filename for the latest csvs
        # Copies file with metadata (creation and modification times)
        blob_container_folder = os.path.join(self.source_location, 'blob_containers')
        
        shutil.copy2(os.path.join(blob_container_folder, 'semif-HighLevelStats.txt'), self.report_folder)

        shutil.copy2(os.path.join(blob_container_folder, f'semif_upload_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'), os.path.join(self.report_folder, f'semif_upload_batch_details_az.csv'))
        shutil.copy2(os.path.join(blob_container_folder, f'semif_cutouts_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'), os.path.join(self.report_folder, f'semif_cutouts_batch_details_az.csv'))
        shutil.copy2(os.path.join(blob_container_folder, f'semif_developed_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'), os.path.join(self.report_folder, f'semif_developed_batch_details_az.csv'))

        lts_location_folder = os.path.join(self.source_location, 'storage_lockers')
        shutil.copy2(os.path.join(lts_location_folder, f'semif_upload_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'), os.path.join(self.report_folder, f'semif_upload_batch_details_lts.csv'))
        shutil.copy2(os.path.join(lts_location_folder, f'semif_cutouts_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'), os.path.join(self.report_folder, f'semif_cutouts_batch_details_lts.csv'))
        shutil.copy2(os.path.join(lts_location_folder, f'semif_developed_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'), os.path.join(self.report_folder, f'semif_developed_batch_details_lts.csv'))
    
    def compose_slack_message(self):
        with open(os.path.join(self.report_folder, 'semif-HighLevelStats.txt'), 'r') as f:
            self.message_blocks.append({
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': ''.join(f.readlines())
                }
            })
        for file in os.listdir(self.report_folder):
            if file.lower().endswith('.png'):
                self.files.append(os.path.join(self.report_folder, file))


    def send_slack_notification(self):
        client = WebClient(token=self.__auth_config_data['slack_api_token'])
        # message = {
        #     "channel": "#semif-datareports",
        #     "blocks": [
        #         {
        #             "type": "section",
        #             "text": {
        #                 "type": "mrkdwn",
        #                 "text": "*Data Report*\n {insert date}"
        #             }
        #         },
        #         {
        #             "type": "divider"
        #         },
        #         {
        #             "type": "section",
        #             "text": {
        #                 "type": "mrkdwn",
        #                 "text": "• Unprocessed 1\n• Processed"
        #             }
        #         },
        #     ]
        # }
        log.info(self.message_blocks)
        message = {
            'channel': self.task_cfg.slack_channel,
            'blocks': self.message_blocks
        }
        try:
            message_response = client.chat_postMessage(**message)
            
            log.info(f"sent slack message to channel - {message_response['channel']}, thread - {message_response['ts']}")

            for file in self.files:
                file_response = client.files_upload_v2(
                    channel=message_response['channel'],
                    file=file,
                    # initial_comment="Here's the attached file",
                    thread_ts=message_response['ts'],
                )
            # print("response 2: ", file_response)
            # if response["ok"]:
            #     print("Message sent successfully")
        except SlackApiError as e:
            print(f"Error: {e}")

    # def generate_developed_batch_graphs(self, csv_file):
    #     type = csv_file.split("_")[-1].replace('.csv', '')

    #     def split_field(row):
    #         parts = row['batch'].split('_')
    #         return pd.Series({'state': parts[0], 'date': parts[1]})
        
    #     df = pd.read_csv(csv_file, index_col=0)
    #     df[['state', 'date']] = df.apply(split_field, axis=1)
    #     df['year'] = pd.to_datetime(df['date']).dt.year
    #     processed_images_by_state = df[df['UnProcessed'] == False].groupby('state')['images'].sum()
    #     processed_images_by_year = df[df['UnProcessed'] == False].groupby('year')['images'].sum()
        
    #     # processed/unprocessed high level stats
    #     plot_data = pd.crosstab(df['state'], df['UnProcessed'])
    #     ax = plot_data.plot(kind='bar', stacked=False, figsize=(10, 6))
    #     plt.title(f'Patterns of UnProcessed True/False by State - {type}')
    #     plt.xlabel('State')
    #     plt.ylabel('Count')
    #     plt.legend(title='UnProcessed')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.report_folder,f'processed_by_states_{type}.png'), dpi=300)

    #     # processed image count by state
    #     plt.figure(figsize=(10, 5))
    #     processed_images_by_state.plot(kind='bar', color='skyblue')
    #     plt.title(f'Number of Processed Images by State - {type}')
    #     plt.xlabel('State')
    #     plt.ylabel('Number of Processed Images')
    #     plt.xticks(rotation=0)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.report_folder,f'processed_images_by_state_{type}.png'), dpi=300)

    #     # processed image count by year
    #     plt.figure(figsize=(10, 5))
    #     processed_images_by_year.plot(kind='bar', color='skyblue')
    #     plt.title(f'Number of Processed Images by Year - {type}')
    #     plt.xlabel('Year')
    #     plt.ylabel('Number of Processed Images')
    #     plt.xticks(rotation=0)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.report_folder,f'processed_images_by_year_{type}.png'), dpi=300)

    #     return
    
    def _cleanup_lts_uploads_csv(self):
        # for duplicate batches (present in multiple lts locations - longterm_storage, GROW_DATA) 
        #     keep records where lts location matches for uploads and developed images
        # for records where some details are missing - raw_count or total_size
        #       get details from azure and update the records
        df = pd.read_csv(os.path.join(self.report_folder, 'semif_upload_batch_details_lts.csv'))
        duplicates = df[df.duplicated('batch', keep=False)]
        filtered_duplicates = duplicates[duplicates['path'] == duplicates['developed_lts_loc']]
        non_duplicates = df[~df.duplicated('batch', keep=False)]
        result = pd.concat([non_duplicates, filtered_duplicates]).sort_index()
        
        # cleanup + compare with azure records - still leaves a few empty batches
        empty_batches = result.loc[((result.raw_count == 0) | (result.totalSizeGiB == 0))]['batch'].tolist()
        az_df = pd.read_csv(os.path.join(self.report_folder, 'semif_upload_batch_details_az.csv'))
        for batch in empty_batches:
            matching_row = az_df[az_df['batch'] == batch]
            if not matching_row.empty:
                result.loc[result['batch'] == batch, ['path', 'raw_count', 'jpg_count', 'totalSizeGiB']] = \
                    matching_row[['path', 'raw_count', 'jpg_count', 'totalSizeGiB']].values
        result.to_csv(os.path.join(self.report_folder, 'semif_upload_batch_details_lts.csv'), index=False)
    
    def _combine_uploads_csv(self, lts_uploads_df, az_uploads_df):
        # merges all records into one csv, removes three empty batches 
        # NC_2022-10-21, NC_2022-11-04, MD_2022-09-14
        combined_df = pd.merge(lts_uploads_df, az_uploads_df, on='batch', how='outer', suffixes=('_lts', '_az'))
        combined_df[~combined_df['batch'].isin(
            lts_uploads_df[((lts_uploads_df.raw_count == 0) & 
                            (lts_uploads_df.jpg_count == 0))]['batch'].tolist())]
        return combined_df

    def _cleanup_developed_duplicates(self, lts_developed_df, az_developed_df):
        # take duplicate batches (present in multiple locations) in lts location
        # remove them from consideration
        # join lts and az developed data
        # return combined df and list of duplicates
        lts_developed_duplicated_batches = lts_developed_df[lts_developed_df['batch'].duplicated()]['batch'].tolist()
        
        lts_developed_df = lts_developed_df[~lts_developed_df['batch'].isin(lts_developed_duplicated_batches)]
        az_developed_df = az_developed_df[az_developed_df['batch'].isin(lts_developed_duplicated_batches)]
        combined_developed_df = pd.merge(lts_developed_df, az_developed_df, on='batch', how='outer', suffixes=('_lts', '_az'))

        # remove empty batch records (according to LTS) 
        # also means az records for those batches are empty
        combined_developed_df = combined_developed_df[~combined_developed_df['batch'].isin(
            lts_developed_df[((lts_developed_df.images == 0) & 
                            (lts_developed_df.metadata == 0) & 
                            (lts_developed_df.meta_masks == 0))]['batch'].tolist())]

        return combined_developed_df, lts_developed_duplicated_batches
    
    def _cleanup_cutouts_duplicates(self, lts_cutouts_df, az_cutouts_df):
        # same as above - takes care of one batch as of now
        lts_cutouts_duplicated_batches = lts_cutouts_df[lts_cutouts_df['batch'].duplicated()]['batch'].tolist()

        lts_cutouts_df = lts_cutouts_df[~lts_cutouts_df['batch'].isin(lts_cutouts_duplicated_batches)]
        az_cutouts_df = az_cutouts_df[~az_cutouts_df['batch'].isin(lts_cutouts_duplicated_batches)]
        combined_df = pd.merge(lts_cutouts_df, az_cutouts_df, on='batch', how='outer', suffixes=('_lts', '_az'))
        combined_cutouts_df = combined_df[~combined_df['batch'].isin(
            lts_cutouts_df[((lts_cutouts_df.jpg_count == 0) & 
                            (lts_cutouts_df.png_count == 0) & 
                            (lts_cutouts_df.json_count == 0) & 
                            (lts_cutouts_df.mask_count == 0))]['batch'].tolist())]
        return combined_cutouts_df, lts_cutouts_duplicated_batches
    


    def generate_actionable_table(self):
        # current implementation ignores duplicates, also has extra variable declarations
        # optimize + update implementation later

        self._cleanup_lts_uploads_csv()
        lts_uploads_df = pd.read_csv(os.path.join(self.report_folder, 'semif_upload_batch_details_lts.csv'))
        az_uploads_df = pd.read_csv(os.path.join(self.report_folder, 'semif_upload_batch_details_az.csv'))
        uploads_df = self._combine_uploads_csv(lts_uploads_df, az_uploads_df)

        az_developed_df = pd.read_csv(os.path.join(self.report_folder, 'semif_developed_batch_details_az.csv'), index_col=0)
        lts_developed_df = pd.read_csv(os.path.join(self.report_folder, 'semif_developed_batch_details_lts.csv'))
        developed_df, lts_developed_duplicated_batches = self._cleanup_developed_duplicates(lts_developed_df, az_developed_df)

        az_cutouts_df = pd.read_csv(os.path.join(self.report_folder, 'semif_cutouts_batch_details_az.csv'), index_col=0)
        lts_cutouts_df = pd.read_csv(os.path.join(self.report_folder, 'semif_cutouts_batch_details_lts.csv'))
        cutouts_df, lts_cutouts_duplicated_batches = self._cleanup_cutouts_duplicates(lts_cutouts_df, az_cutouts_df)

        records = []
        for _, row in uploads_df.iterrows():
            name_splits = row['batch'].split("_")
            if row['path_lts']:
                if row['path_lts'] != 'azure':
                    upload_lts = True
                else:
                    upload_lts = False
            else:
               upload_lts = False
            
            
            records.append({
               "batch": row['batch'],
               "developed_lts": False,
               "developed_azure": False,
               "preprocessed": row['IsPreprocessed_lts'] if row['IsPreprocessed_lts'] is not None else row['IsPreprocessed_az'],
               "processed": False,
               "state": name_splits[0],
               "date": name_splits[1],
               "developed_jpgs": 0,
               "upload_raws": row['raw_count_lts'] if row['path_lts'] else row['raw_count_az'],
               "upload_lts": upload_lts,
               "upload_azure": True if row['path_az'] else False,
               "cutouts_lts": None,
               "cutouts_azure": None,
               "bbot_version": row['version']
            })
        
        pd.DataFrame(records).to_csv(os.path.join(self.report_folder, f"actionable_items_{datetime.now().strftime("%Y%m%d")}.csv"))


        

def main(cfg: DictConfig) -> None:
    """Main function to execute report generation and sending it to slack."""
    report = Report(cfg)

    report.copy_relevant_files()
    report.generate_actionable_table()

    # report._cleanup_lts_uploads_csv()
    # report.generate_developed_batch_graphs(os.path.join(report.report_folder, 'semif_developed_batch_details_az.csv'))
    # report.generate_developed_batch_graphs(os.path.join(report.report_folder, 'semif_developed_batch_details_lts.csv'))
    # log.info('generating report')

    # report.compose_slack_message()
    # log.info('sending slack notification')
    # report.send_slack_notification()
