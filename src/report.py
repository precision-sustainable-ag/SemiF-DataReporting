import logging
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from utils.utils import read_yaml, _get_bbot_version

log = logging.getLogger(__name__)


class Report:
    def __init__(self, cfg: DictConfig) -> None:
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.task_cfg = cfg.report
        self.source_location = cfg.paths.data_dir
        self.parent_report_folder = cfg.paths.report
        self.report_folder = None
        self.summary_stats = {}
        self.bbot_versions = cfg.bbot_versions

    def copy_relevant_files(self):
        # get relevant blobcontainers files
        # pass in the filename for the latest csvs
        # Copies file with metadata (creation and modification times)

        os.makedirs(os.path.join(self.parent_report_folder,
                                 datetime.now().strftime("%Y-%m-%d")),
                    exist_ok=True)
        self.report_folder = os.path.join(self.parent_report_folder,
                                          datetime.now().strftime("%Y-%m-%d"))
        blob_container_folder = os.path.join(self.source_location,
                                             'blob_containers')

        shutil.copy2(
            os.path.join(blob_container_folder, 'semif-HighLevelStats.txt'),
            self.report_folder)

        shutil.copy2(os.path.join(blob_container_folder,
                                  f'semif_upload_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'),
                     os.path.join(self.report_folder,
                                  f'semif_upload_batch_details_az.csv'))
        shutil.copy2(os.path.join(blob_container_folder,
                                  f'semif_cutouts_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'),
                     os.path.join(self.report_folder,
                                  f'semif_cutouts_batch_details_az.csv'))
        shutil.copy2(os.path.join(blob_container_folder,
                                  f'semif_developed_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'),
                     os.path.join(self.report_folder,
                                  f'semif_developed_batch_details_az.csv'))

        lts_location_folder = os.path.join(self.source_location,
                                           'storage_lockers')
        shutil.copy2(os.path.join(lts_location_folder,
                                  f'semif_upload_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'),
                     os.path.join(self.report_folder,
                                  f'semif_upload_batch_details_lts.csv'))
        shutil.copy2(os.path.join(lts_location_folder,
                                  f'semif_cutouts_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'),
                     os.path.join(self.report_folder,
                                  f'semif_cutouts_batch_details_lts.csv'))
        shutil.copy2(os.path.join(lts_location_folder,
                                  f'semif_developed_batch_details_{datetime.now().strftime("%Y%m%d")}.csv'),
                     os.path.join(self.report_folder,
                                  f'semif_developed_batch_details_lts.csv'))

    def send_slack_notification(self, message_blocks, files):
        client = WebClient(token=self.__auth_config_data['slack_api_token'])
        message = {
            'channel': self.task_cfg.slack_channel,
            'blocks': message_blocks
        }
        try:
            message_response = client.chat_postMessage(**message)

            log.info(
                f"sent slack message to channel - {message_response['channel']}, thread - {message_response['ts']}")

            for file in files:
                file_response = client.files_upload_v2(
                    channel=message_response['channel'],
                    file=file,
                    # initial_comment="Here's the attached file",
                    thread_ts=message_response['ts'],
                )
        except SlackApiError as e:
            print(f"Error: {e}")

    def _cleanup_lts_uploads_csv(self):
        # for duplicate batches (present in multiple lts locations - longterm_storage, GROW_DATA) 
        #     keep records where lts location matches for uploads and developed images
        # for records where some details are missing - raw_count or total_size
        #       get details from azure and update the records
        df = pd.read_csv(os.path.join(self.report_folder,
                                      'semif_upload_batch_details_lts.csv'))
        duplicates = df[df.duplicated('batch', keep=False)]
        filtered_duplicates = duplicates[
            duplicates['path'] == duplicates['developed_lts_loc']]
        non_duplicates = df[~df.duplicated('batch', keep=False)]
        result = pd.concat([non_duplicates, filtered_duplicates]).sort_index()

        # cleanup + compare with azure records - still leaves a few empty batches
        empty_batches = \
            result.loc[((result.raw_count == 0) | (result.totalSizeGiB == 0))][
                'batch'].tolist()
        az_df = pd.read_csv(os.path.join(self.report_folder,
                                         'semif_upload_batch_details_az.csv'))
        for batch in empty_batches:
            matching_row = az_df[az_df['batch'] == batch]
            if not matching_row.empty:
                result.loc[
                    result['batch'] == batch, ['path', 'raw_count', 'jpg_count',
                                               'totalSizeGiB']] = \
                    matching_row[['path', 'raw_count', 'jpg_count',
                                  'totalSizeGiB']].values
        result.to_csv(os.path.join(self.report_folder,
                                   'semif_upload_batch_details_lts.csv'),
                      index=False)

    def _combine_uploads_csv(self, lts_uploads_df, az_uploads_df):
        # merges all records into one csv, removes three empty batches 
        # NC_2022-10-21, NC_2022-11-04, MD_2022-09-14
        combined_df = pd.merge(lts_uploads_df, az_uploads_df, on='batch',
                               how='outer', suffixes=('_lts', '_az'))
        combined_df = combined_df[~combined_df['batch'].isin(
            lts_uploads_df[((lts_uploads_df.raw_count == 0) &
                            (lts_uploads_df.jpg_count == 0))][
                'batch'].tolist())]
        combined_df = combined_df.replace(np.nan, None)
        return combined_df

    def _developed_deduplication_logic(self, group):
        # logic - whichever path has max sum of (images+metadata+meta_masks)
        # is kept
        if len(group) == 2:
            return group.loc[(group[['images', 'metadata', 'meta_masks']].sum(
                axis=1)).idxmax()]
        elif len(group) > 2:
            log.warning(f"{group['batch'].tolist()[0]} -developed - has more "
                     f"than 2 duplicates")
            return None

    def _cleanup_developed_duplicates(self, lts_developed_df, az_developed_df):
        # take duplicate batches (present in multiple locations) in lts location
        # remove one of the two from consideration
        # join lts and az developed data
        # return combined df and list of duplicates
        lts_developed_duplicated_batches = \
            lts_developed_df[lts_developed_df['batch'].duplicated()][
                'batch'].tolist()

        # duplicate records saved separately
        lts_developed_duplicates = lts_developed_df[
            lts_developed_df['batch'].isin(lts_developed_duplicated_batches)]

        # deduplicate batch records using _developed_deduplication_logic
        # merge with other records -> lts_developed_df
        cleaned_lts_duplicates_df = lts_developed_df.groupby('batch',
                                                             as_index=False).apply(
            self._developed_deduplication_logic).reset_index(drop=True)
        lts_developed_df = pd.concat([lts_developed_df[
                                          ~lts_developed_df['batch'].duplicated(
                                              keep=False)],
                                      cleaned_lts_duplicates_df
                                      ]).drop_duplicates().reset_index(
            drop=True)
        # az doesn't have the duplicate records
        # lts_developed_df = lts_developed_df[
        #     ~lts_developed_df['batch'].isin(lts_developed_duplicated_batches)]
        # az_developed_df = az_developed_df[
        #     az_developed_df['batch'].isin(lts_developed_duplicated_batches)]

        combined_developed_df = pd.merge(lts_developed_df, az_developed_df,
                                         on='batch', how='outer',
                                         suffixes=('_lts', '_az'))

        # remove empty batch records (according to LTS) 
        # also means az records for those batches are empty
        combined_developed_df = combined_developed_df[
            ~combined_developed_df['batch'].isin(
                lts_developed_df[((lts_developed_df.images == 0) &
                                  (lts_developed_df.metadata == 0) &
                                  (lts_developed_df.meta_masks == 0))][
                    'batch'].tolist())]
        combined_developed_df = combined_developed_df.replace(np.nan, None)
        combined_developed_df.dropna(how='all', inplace=True)
        return combined_developed_df, lts_developed_duplicates, lts_developed_duplicated_batches

    def _cleanup_cutouts_duplicates(self, lts_cutouts_df, az_cutouts_df):
        # same as above - takes care of one batch as of now
        lts_cutouts_duplicated_batches = \
            lts_cutouts_df[lts_cutouts_df['batch'].duplicated()][
                'batch'].tolist()

        lts_cutouts_df = lts_cutouts_df[
            ~lts_cutouts_df['batch'].isin(lts_cutouts_duplicated_batches)]
        az_cutouts_df = az_cutouts_df[
            ~az_cutouts_df['batch'].isin(lts_cutouts_duplicated_batches)]
        combined_cutouts_df = pd.merge(lts_cutouts_df, az_cutouts_df,
                                       on='batch',
                                       how='outer', suffixes=('_lts', '_az'))
        combined_cutouts_df = combined_cutouts_df[
            ~combined_cutouts_df['batch'].isin(
                lts_cutouts_df[((lts_cutouts_df.jpg_count == 0) &
                                (lts_cutouts_df.png_count == 0) &
                                (lts_cutouts_df.json_count == 0) &
                                (lts_cutouts_df.mask_count == 0))][
                    'batch'].tolist())]
        combined_cutouts_df = combined_cutouts_df.replace(np.nan, None)
        return combined_cutouts_df, lts_cutouts_duplicated_batches

    def _generate_summary_stats(self, uploads_df, developed_df, cutouts_df):
        actionable_df = pd.read_csv(os.path.join(self.report_folder,
                                                 'batch_details.csv'),
                                    index_col=0)
        processed_counts = pd.crosstab(actionable_df['state'],
                                       actionable_df['processed'])
        # processed_counts.plot(kind='bar', width=0.8)
        ax = processed_counts.plot(kind='bar', stacked=False, figsize=(10, 6))
        plt.title(f'Processed vs Unprocessed batches by State')
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.legend(title='Processed')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.report_folder, f'processed_by_states.png'),
            dpi=300)
        self.summary_stats['files'] = [os.path.join(self.report_folder,
                                                    f'processed_by_states.png')]
        self.summary_stats['totalSizeTB'] = {
            'uploads': (uploads_df['totalSizeGiB_lts'].fillna(
                uploads_df['totalSizeGiB_az']).sum()) / 1024,
            'developed': (developed_df['ImagesFolderSizeGiB_lts'].fillna(
                developed_df['ImagesFolderSizeGiB_az']).sum() +
                          developed_df['MetadataFolderSizeGiB_lts'].fillna(
                              developed_df['MetadataFolderSizeGiB_az']).sum() +
                          developed_df['MetaMasksFolderSizeGiB_lts'].fillna(
                              developed_df['MetaMasksFolderSizeGiB_az']).sum()
                          ) / 1024,
            'cutouts': (cutouts_df['jpg_size_gib_lts'].fillna(
                cutouts_df['jpg_size_gib_az']).sum() +
                        cutouts_df['png_size_gib_lts'].fillna(
                            cutouts_df['png_size_gib_az']).sum() +
                        cutouts_df['json_size_gib_lts'].fillna(
                            cutouts_df['json_size_gib_az']).sum() +
                        cutouts_df['mask_size_gib_lts'].fillna(
                            cutouts_df['mask_size_gib_az']).sum()) / 1024
        }

    def generate_actionable_table(self):
        # current implementation ignores duplicates, also has extra variable declarations
        # optimize + update implementation later

        self._cleanup_lts_uploads_csv()
        lts_uploads_df = pd.read_csv(os.path.join(self.report_folder,
                                                  'semif_upload_batch_details_lts.csv'))  # nopep8
        az_uploads_df = pd.read_csv(os.path.join(self.report_folder,
                                                 'semif_upload_batch_details_az.csv'))  # nopep8
        uploads_df = self._combine_uploads_csv(lts_uploads_df, az_uploads_df)

        az_developed_df = pd.read_csv(os.path.join(self.report_folder,
                                                   'semif_developed_batch_details_az.csv'),
                                      index_col=0)
        lts_developed_df = pd.read_csv(os.path.join(self.report_folder,
                                                    'semif_developed_batch_details_lts.csv'))  # nopep8
        developed_df, lts_developed_duplicates_df, lts_developed_duplicated_batches = self._cleanup_developed_duplicates(
            lts_developed_df, az_developed_df)

        # save duplicates for developed
        lts_developed_duplicates_df.to_csv(os.path.join(self.report_folder,
                                                        'semif_developed_duplicates_lts.csv'))  # nopep8
        az_cutouts_df = pd.read_csv(os.path.join(self.report_folder,
                                                 'semif_cutouts_batch_details_az.csv'),
                                    index_col=0)
        lts_cutouts_df = pd.read_csv(os.path.join(self.report_folder,
                                                  'semif_cutouts_batch_details_lts.csv'))  # nopep8
        cutouts_df, lts_cutouts_duplicated_batches = self._cleanup_cutouts_duplicates(
            lts_cutouts_df, az_cutouts_df)

        # do we take upload_raws, developed_jpgs, ..
        # from azure if that detail is not present in lts?
        records = {}
        for _, row in uploads_df.iterrows():
            name_splits = row['batch'].split("_")
            # since _cleanup is done by taking batch details from azure
            # (where lts is empty), it means these batches are in az but not
            # in lts
            if row['path_lts']:
                if row['path_lts'] != 'azure':
                    upload_lts = True
                else:
                    upload_lts = False
            else:
                upload_lts = False
            records[row['batch']] = {
                "batch": row['batch'],
                "developed_lts": False,
                "developed_azure": False,
                "preprocessed": row['IsPreprocessed_lts']
                if row['IsPreprocessed_lts'] is not None
                else row['IsPreprocessed_az'],
                "processed": False,
                "state": name_splits[0],
                "date": name_splits[1],
                "developed_jpgs": None,
                "upload_raws": row['raw_count_lts'] if row['path_lts']
                else row['raw_count_az'],
                "upload_lts": upload_lts,
                "upload_azure": True if row['path_az'] else False,
                "cutouts_lts": None,
                "cutouts_azure": None,
                "bbot_version": _get_bbot_version(self.bbot_versions,
                                                  name_splits[0],
                                                  name_splits[1]),
            }
        upload_lts = None

        for _, row in developed_df.iterrows():
            if row['batch'] in records:
                records[row['batch']]['developed_lts'] = True if row[
                    'path_lts'] else False
                records[row['batch']]['developed_azure'] = True if row[
                    'path_az'] else False
                records[row['batch']]['processed'] = not \
                    row['UnProcessed_lts'] if row[
                                                  'UnProcessed_lts'] is not None \
                    else not row['UnProcessed_az']
                records[row['batch']]['developed_jpgs'] = row[
                    'images_lts'] if \
                    row['images_lts'] else row['images_az']
            else:
                log.info(
                    f"{row['batch']} - developed found, upload missing")
                name_splits = row['batch'].split("_")
                records[row['batch']] = {
                    "batch": row['batch'],
                    "developed_lts": True if row['path_lts'] else False,
                    "developed_azure": True if row['path_az'] else False,
                    "preprocessed": None,
                    "processed": not row['UnProcessed_lts'] if
                    row['UnProcessed_lts'] is not None
                    else not row['UnProcessed_az'],
                    "state": name_splits[0],
                    "date": name_splits[1],
                    "developed_jpgs": row['images_lts'] if row['images_lts']
                    else row['images_az'],
                    "upload_raws": None,
                    "upload_lts": None,
                    "upload_azure": None,
                    "cutouts_lts": None,
                    "cutouts_azure": None,
                    "bbot_version": _get_bbot_version(self.bbot_versions,
                                                      name_splits[0], name_splits[1])
                }

        for _, row in cutouts_df.iterrows():
            # avoiding cutouts duplicate batches for now - since there is
            # only one
            if row['batch'] not in lts_cutouts_duplicated_batches:
                if row['batch'] in records:
                    records[row['batch']]['cutouts_lts'] = True if row[
                        'path_lts'] else False
                    records[row['batch']]['cutouts_azure'] = True if row[
                        'path_az'] else False
                else:
                    log.info(f"{row['batch']} - cutouts found, rest missing")
                    name_splits = row['batch'].split("_")
                    records[row['batch']] = {
                        "batch": row['batch'],
                        "developed_lts": None,
                        "developed_azure": None,
                        "preprocessed": None,
                        "processed": None,
                        "state": name_splits[0],
                        "date": name_splits[1],
                        "developed_jpgs": None,
                        "upload_raws": None,
                        "upload_lts": None,
                        "upload_azure": None,
                        "cutouts_lts": True if row['path_lts'] else False,
                        "cutouts_azure": True if row['path_az'] else False,
                        "bbot_version": _get_bbot_version(self.bbot_versions,
                                                      name_splits[0], name_splits[1])
                    }
        # handle odd cases where lts has uploads and az has developed or vice
        # versa
        for batch_name, batch_info in records.items():
            if (batch_info['upload_lts'] or batch_info['upload_azure']) and (
                    batch_info['developed_lts'] or
                    batch_info['developed_azure']):
                batch_info['preprocessed'] = True
            else:
                batch_info['preprocessed'] = False

        records = [v for k, v in records.items()]
        pd.DataFrame(records).to_csv(
            os.path.join(self.report_folder, f"batch_details.csv"))
        self._generate_summary_stats(uploads_df, developed_df, cutouts_df)
        # TODO: deal with cutouts duplicate records
        return

    def generate_actionable_message(self):
        message_blocks = [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*SemiF - Actionable items Report* - "
                            f"{datetime.now().strftime('%m/%d/%Y')}"
                }
            },
            {
                'type': 'divider'
            }
        ]
        actionable_df = pd.read_csv(
            os.path.join(self.report_folder, 'batch_details.csv'),
            index_col=0)
        uploads_az_not_lts = actionable_df[(actionable_df['upload_azure']) & (
                actionable_df['upload_lts'] == False)].shape[0]
        developed_az_not_lts = actionable_df[
            (actionable_df['developed_azure']) & (
                    actionable_df['developed_lts'] == False)].shape[0]
        cutouts_az_not_lts = actionable_df[(actionable_df['cutouts_azure']) & (
                actionable_df['cutouts_lts'] == False)].shape[0]
        message_blocks.extend([
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"Not preprocessed: "
                            f"*{actionable_df['preprocessed'].eq(False).sum()}*"
                            f" batches.\n"
                            f"Missing preprocessing info: "
                            f"*{actionable_df['preprocessed'].isnull().sum()}* "
                            f"batches"
                },
            },
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"Not processed: "
                            f"*{actionable_df['processed'].eq(False).sum()}* "
                            f"batches.\n"
                            f"Missing processing info: "
                            f"*{actionable_df['processed'].isnull().sum()}* "
                            f"batches."

                },
            },
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*Batches present in AZ but not in LTS:*\n"
                            f"• semif-uploads: {uploads_az_not_lts}\n"
                            f"• semif-developed-images: "
                            f"{developed_az_not_lts}\n"
                            f"• semif-cutouts: {cutouts_az_not_lts}"
                },
            },
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"Uploaded images: "
                            f"*{actionable_df['upload_raws'].sum()}*\n"
                            f"Developed images: "
                            f"*{actionable_df['developed_jpgs'].sum()}*"
                },
            }
        ])

        files = [os.path.join(self.report_folder, 'batch_details.csv'),
                 os.path.join(self.report_folder,
                              'semif_developed_duplicates_lts.csv')]

        return message_blocks, files

    def generate_summary_message(self):
        message_blocks = [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*SemiF - Data Report* - "
                            f"{datetime.now().strftime('%m/%d/%Y')}"
                }
            },
            {
                'type': 'divider'
            }
        ]
        md_table = "```" \
                   "| Category   | Total Size (TB) |\n" \
                   "|------------|-----------------|\n"
        for category, size in self.summary_stats['totalSizeTB'].items():
            md_table += f"| {category.capitalize()} | {round(size, 2)} |\n"
        md_table += '\n```'
        message_blocks.append({
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': md_table
            }
        })
        return message_blocks, self.summary_stats['files']


def main(cfg: DictConfig) -> None:
    """Main function to execute report generation and sending it to slack."""
    report = Report(cfg)

    report.copy_relevant_files()
    report.generate_actionable_table()
    message, files = report.generate_summary_message()
    report.send_slack_notification(message, files)

    message, files = report.generate_actionable_message()
    report.send_slack_notification(message, files)
