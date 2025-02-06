import logging
import typing
import shutil
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from prettytable import PrettyTable

from utils.utils import read_yaml, _get_bbot_version

log = logging.getLogger(__name__)


class Report:
    """
    Class to generate a report with components to send to Slack, and other
    helper functions.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the report class.
        Args:
            cfg (DictConfig): main hydra configuration object
        """
        self.__auth_config_data = read_yaml(cfg.paths.pipeline_keys)
        self.task_cfg = cfg.report
        self.source_location = cfg.paths.data_dir
        self.parent_report_folder = cfg.paths.report
        self.report_folder = None
        self.summary_stats = {}
        self.bbot_versions = cfg.bbot_versions

    def copy_relevant_files(self) -> None:
        """
        get relevant blobcontainers files
        pass in the filename for the latest csvs
        Copies file with metadata (creation and modification times)
        """

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

    def send_slack_notification(self, message_blocks: typing.List[typing.Dict],
                                files: typing.List) -> None:
        """
        Function to send Slack notification
        Args:
            message_blocks (List[Dict]): list of message_blocks
            files (List): list of files (filepaths)
        """
        client = WebClient(token=self.__auth_config_data['slack_api_token'])
        message = {
            'channel': self.task_cfg.slack_channel,
            'blocks': message_blocks
        }
        try:
            message_response = client.chat_postMessage(**message)
            if files:
                for file in files:
                    file_response = client.files_upload_v2(
                        channel=message_response['channel'],
                        file=file,
                        thread_ts=message_response['ts'],
                    )
        except SlackApiError as e:
            log.info(f"Error: {e}")

    def _cleanup_lts_uploads_csv(self):
        """
        Function to clean up lts uploads csv
        """

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

    def _combine_uploads_csv(self, lts_uploads_df: pd.DataFrame,
                             az_uploads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine details about uploads from lts and azure blobs and remove
        empty batches
        Args:
            lts_uploads_df (pd.DataFrame): dataframe of lts uploads
            az_uploads_df (pd.DataFrame): dataframe of azure uploads
        """

        combined_df = pd.merge(lts_uploads_df, az_uploads_df, on='batch',
                               how='outer', suffixes=('_lts', '_az'))
        combined_df = combined_df[~combined_df['batch'].isin(
            lts_uploads_df[((lts_uploads_df.raw_count == 0) &
                            (lts_uploads_df.jpg_count == 0))][
                'batch'].tolist())]
        combined_df = combined_df.replace(np.nan, None)
        return combined_df

    def _developed_deduplication_logic(self, group):
        """
        Logic to select a batch when multiple lts locations have developed
        images
        Args:
            group: dataframe group with same batch name
        """

        # logic - whichever path has max sum of (images+metadata+meta_masks)
        # is kept
        if len(group) == 2:
            return group.loc[(group[['images', 'metadata', 'meta_masks']].sum(
                axis=1)).idxmax()]
        elif len(group) > 2:
            log.warning(f"{group['batch'].tolist()[0]} -developed - has more "
                        f"than 2 duplicates")
            return None

    def _cleanup_developed_duplicates(self, lts_developed_df: pd.DataFrame,
                                      az_developed_df: pd.DataFrame) -> (
            pd.DataFrame, pd.DataFrame, typing.List[str]):
        """
        Function to clean up duplicate batch entries from lts and azure
        Args:
            lts_developed_df (pd.DataFrame): dataframe of developed images in lts
            az_developed_df (pd.DataFrame): dataframe of developed images in
            azure
        """
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

    def _cleanup_cutouts_duplicates(self, lts_cutouts_df: pd.DataFrame,
                                    az_cutouts_df: pd.DataFrame) -> (
            pd.DataFrame, typing.List[str]):
        """
        Function to clean up cutouts duplicates from lts and azure
        Args:
            lts_cutouts_df (pd.DataFrame): dataframe of lts cutouts
            az_cutouts_df (pd.DataFrame): dataframe of azure cutouts
        """
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

    def _generate_summary_stats(self, uploads_df: pd.DataFrame, developed_df:
    pd.DataFrame, cutouts_df: pd.DataFrame) -> None:
        """
        Function to generate summary stats from actionable table
        Also generates the needed graph
        Args:
             uploads_df (pd.DataFrame): dataframe of uploads
             developed_df (pd.DataFrame): dataframe of developed
             cutouts_df (pd.DataFrame): dataframe of cutouts
        """
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

    def generate_actionable_table(self) -> None:
        """
        Main function to generate actionable table
        Runs multiple internal functions to combine uploads csvs, developed
        csvs, cutouts csvs
        """

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
                                                      name_splits[0],
                                                      name_splits[1])
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
                                                          name_splits[0],
                                                          name_splits[1])
                    }
        # handle odd cases where lts has uploads and az has developed or vice
        # versa
        for batch_name, batch_info in records.items():
            if (batch_info['upload_lts'] or batch_info['upload_azure']) and (
                    batch_info['developed_lts'] or
                    batch_info['developed_azure']):
                batch_info['preprocessed'] = True
            elif batch_info['developed_jpgs'] and batch_info[
                'developed_jpgs'] > 0:
                batch_info['preprocessed'] = True
            else:
                batch_info['preprocessed'] = False

        records = [v for k, v in records.items()]
        pd.DataFrame(records).to_csv(
            os.path.join(self.report_folder, f"batch_details.csv"))
        self._generate_summary_stats(uploads_df, developed_df, cutouts_df)
        # TODO: deal with cutouts duplicate records
        return

    def generate_actionable_message(self) -> (typing.List[typing.Dict],
                                              typing.List[str]):
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

    def generate_summary_message(self) -> (
            typing.List[typing.Dict], typing.List[str]):
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


class DBQuery:
    """
    Class to connect to the database and run queries.
    Also saves the stats as csv files.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Class initialization method.
        Connects to the database and initializes the db cursor.
        Args:
            cfg (DictConfig): Main hydra config object.
        """
        self.report_folder = os.path.join(cfg.paths.report,
                                          datetime.now().strftime("%Y-%m-%d"))
        self.txt_db_report = os.path.join(self.report_folder,
                                          'general_dist_stats.txt')
        # initialize the report in case the file already exists
        with open(self.txt_db_report, 'w') as file:
            file.write('')
        self.db_cfg = cfg.report.db
        self.db_name = self.db_cfg['path']

        start_time = datetime.now()
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        log.info(f"Database connection time: {datetime.now() - start_time}")

        self.dev_img_table = self.db_cfg['dev_imgs']
        self.cutout_table = self.db_cfg['cutouts']

    def __del__(self):
        """
        Class destructor method.
        Closes the database connection.
        """
        if self.connection:
            self.connection.commit()
            self.connection.close()
            self.connection = None

    def _execute_query(self, query: str) -> (typing.List[typing.List],
                                             typing.List[str]):
        """
        Internal function to execute a query.
        Args:
            query (str): Query string.
        returns:
        rows and column names of the query.
        """
        try:
            start_time = datetime.now()
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            column_names = [description[0] for description in
                            self.cursor.description]
            log.info(f"Query time: {datetime.now() - start_time}, {query}")
            return rows, column_names
        except sqlite3.Error as e:
            log.error(f"Query failed: {e}")
            return None, None

    @staticmethod
    def _create_table(rows: typing.List, column_names: typing.List, title:
    str) -> PrettyTable:
        """
        internal static method to create pretty tables from rows, columns and title.
        Args:
            rows (list): List of rows.
            column_names (list): List of column names.
            title (str): Table title.
        """

        table = PrettyTable()
        table.title = title
        if rows and column_names:
            table.field_names = column_names
            table.add_rows(rows)
        return table

    @staticmethod
    def _collate_categories(rows: typing.List) -> typing.List:
        """
        The database tables have different categories than needed for stats.
        Internal static function to combine category names to weed crops,
        cash crops, and cover crops.
        Args:
            rows (list): List of rows.
        Returns:
            rows (list): List of modified rows, with combined counts.
        """
        # collate categories into weeds, cash crops and cover crops
        modified_rows = {
            'weed crops': 0,
            'cash crops': 0,
            'cover crops': 0,
        }
        for row in rows:
            if 'weed' in row[0].lower():
                modified_rows['weed crops'] += row[1]
            elif 'cash' in row[0].lower():
                modified_rows['cash crops'] += row[1]
            elif 'cover' in row[0].lower():
                modified_rows['cover crops'] += row[1]
            else:
                if row[0] not in modified_rows:
                    modified_rows[row[0]] = row[1]
                else:
                    modified_rows[row[0]] += row[1]
        rows = [[k, v] for k, v in modified_rows.items()]
        return rows

    def _append_general_dist_txt(self, tables, stat_type):
        with open(self.txt_db_report, 'a') as f:
            f.write(f"Stats for {stat_type}\n\n")
            f.writelines([x+'\n' for x in tables])
            f.write('\n\n--------------------------------------\n\n')
        return

    def _save_csv(self, rows: typing.List, cols: typing.List, subfolder: str,
                  stat_type: str) -> None:
        """
        Function to save csv files for individual stats.
        Args:
            rows (list): List of rows.
            cols (list): List of column names.
            subfolder (str): developed | cutouts | primary_cutouts.

        """
        os.makedirs(os.path.join(self.report_folder, subfolder), exist_ok=True)
        file_name = os.path.join(self.report_folder, subfolder,
                                 f"{stat_type}.csv")
        pd.DataFrame(rows, columns=cols).to_csv(file_name, index=False)

    def fullsized_data(self) -> None:
        """
        Function to execute all queries to generate stats for developed images,
        save the csvs, and generate message block to send to slack
        """

        rows, cols = self._execute_query(f"""
        select count(*) as count from {self.dev_img_table};
        """)
        total_count_table = self._create_table(rows, cols,
                                               "Total Processed Images")

        self._save_csv(rows, cols, 'developed_images', 'total_count')

        rows, cols = self._execute_query(f"""
            select common_name, count(distinct image_id) as images from (
                select json_extract(category.value, '$.common_name') AS common_name, image_id
                from {self.dev_img_table}, json_each(categories) AS category
                )
                group by common_name
                order by images desc;
            """)
        total_by_common_name = self._create_table(rows, cols,
                                                  "Total Processed Images by Common Name")
        self._save_csv(rows, cols, 'developed_images', 'total_by_common_name')

        rows, cols = self._execute_query(f"""
            select category, count(distinct image_id) as images from (
                select json_extract(category_each.value, '$.category') as category, image_id
                from {self.dev_img_table},json_each(categories) as category_each)
                group by category
                order by images desc;
            """)
        rows = self._collate_categories(rows)
        total_by_category = self._create_table(rows, cols,
                                               "Total Processed Images by Category")
        self._save_csv(rows, cols, 'developed_images', 'total_by_category')

        rows, cols = self._execute_query(f"""
            select substr(batch_id, 1, instr(batch_id, '_') - 1) as location, 
            count(*) as count 
            from {self.dev_img_table}
            group by location
            order by location;
            """)
        total_by_location = self._create_table(rows, cols,
                                               "Total Processed Images by Location")
        self._save_csv(rows, cols, 'developed_images', 'total_by_location')

        rows, cols = self._execute_query(f"""
            select substr(datetime, 1, 4) as year, count(*) as count
            from {self.dev_img_table}
            group by year
            order by year desc;
            """)
        total_by_year = self._create_table(rows, cols,
                                           "Total Processed Images by Year")
        self._save_csv(rows, cols, 'developed_images', 'total_by_year')

        tables = [x.get_string() for x in [total_count_table, total_by_common_name,
                          total_by_category, total_by_location,
                           total_by_year]]
        self._append_general_dist_txt(tables, 'developed images')

        return

    def cutout_data(self) -> None:
        """
        Function to execute all queries to generate stats for cutouts,
        save the csvs, and generate message block to send to slack.
        """
        rows, cols = self._execute_query(f"""
        select count(*) from {self.cutout_table};
        """)
        total_count_table = self._create_table(rows, cols, "Total Cutouts")
        self._save_csv(rows, cols, 'cutouts', 'total_count')

        rows, cols = self._execute_query(f"""
        select json_extract(category, '$.common_name') as common_name, count(*) as
        count from {self.cutout_table}
        group by common_name
        order by count desc;
        """)
        total_by_common_name = self._create_table(rows, cols, "Total Cutouts "
                                                              "by Common Name")
        self._save_csv(rows, cols, 'cutouts', 'total_by_common_name')

        rows, cols = self._execute_query(f"""
        select json_extract(category, '$.category') as categories, count(*) as count
        from {self.cutout_table}
        group by categories
        order by count desc;
        """)
        rows = self._collate_categories(rows)
        total_by_category = self._create_table(rows, cols, "Total Cutouts by "
                                                           "Category")
        self._save_csv(rows, cols, 'cutouts', 'total_by_category')

        rows, cols = self._execute_query(f"""
        select substr(batch_id, 1, instr(batch_id, '_') - 1) as location, count(*) as
        count from {self.cutout_table}
        group by location
        order by location desc;
        """)
        total_by_location = self._create_table(rows, cols,
                                               "Total Cutouts by Location")
        self._save_csv(rows, cols, 'cutouts', 'total_by_location')

        rows, cols = self._execute_query(f"""
        select substr(datetime, 1, 4) as year, count(*) as count
        from {self.cutout_table}
        group by year
        order by year desc;
        """)
        total_by_year = self._create_table(rows, cols, "Total Cutouts by Year")
        self._save_csv(rows, cols, 'cutouts', 'total_by_year')

        rows, cols = self._execute_query(f"""
        select
            case
                when json_extract(cutout_props, '$.bbox_area_cm2') between 0 and 0.1
                    then 'very small'
                when json_extract(cutout_props, '$.bbox_area_cm2') between 0.1 and 1
                    then 'small'
                when json_extract(cutout_props, '$.bbox_area_cm2') between 1 and 10
                    then 'medium'
                when json_extract(cutout_props, '$.bbox_area_cm2') between 10 and 100
                    then 'medium large'
                when json_extract(cutout_props, '$.bbox_area_cm2') between 100 and 1000
                    then 'large'
                when json_extract(cutout_props, '$.bbox_area_cm2') between 1000 and 1000000
                    then 'very large'
                else json_extract(cutout_props, '$.bbox_area_cm2')
            end as bbox_area,
            json_extract(semif_cutouts.category, '$.common_name') as common_name,
            json_extract(cutout_props, '$.is_primary') as is_primary,
            count(*) as count
        from {self.cutout_table}
        group by common_name, bbox_area, is_primary;
        """)
        area_by_common_name = self._create_table(rows, cols,"Area by Common Name")
        self._save_csv(rows, cols, 'cutouts', 'area_by_common_name')

        tables = [x.get_string() for x in
                  [total_count_table, total_by_common_name,
                   total_by_category, total_by_location,
                   total_by_year, area_by_common_name]]
        self._append_general_dist_txt(tables, 'cutouts')

        return

    def primary_cutout_data(self) -> None:
        """
        Function to execute all queries to generate stats for cutouts,
        save the csvs, and generate message block to send to slack.
        """
        rows, cols = self._execute_query(f"""
                select count(*) from {self.cutout_table}
                where json_extract(cutout_props, '$.is_primary') = true;
                """)
        total_count_table = self._create_table(rows, cols,
                                               "Total Primary Cutouts")
        self._save_csv(rows, cols, 'primary_cutouts', 'total_count')

        rows, cols = self._execute_query(f"""
                select json_extract(category, '$.common_name') as common_name, count(*) as
                count from {self.cutout_table}
                where json_extract(cutout_props, '$.is_primary') = true
                group by common_name
                order by count desc;
                """)
        total_by_common_name = self._create_table(rows, cols,
                                                  "Total Primary Cutouts by Common Name")
        self._save_csv(rows, cols, 'primary_cutouts', 'total_by_common_name')

        rows, cols = self._execute_query(f"""
                select json_extract(category, '$.category') as category, count(*) as count
                from {self.cutout_table}
                where json_extract(cutout_props, '$.is_primary') = true
                group by category
                order by count desc;
                """)
        rows = self._collate_categories(rows)
        total_by_category = self._create_table(rows, cols,
                                               "Total Primary Cutouts by Category")
        self._save_csv(rows, cols, 'primary_cutouts', 'total_by_category')

        rows, cols = self._execute_query(f"""
                select substr(batch_id, 1, instr(batch_id, '_') - 1) as location, count(*) as
                count from {self.cutout_table}
                where json_extract(cutout_props, '$.is_primary') = true
                group by location
                order by location desc;
                """)
        total_by_location = self._create_table(rows, cols,
                                               "Total Primary Cutouts by Location")
        self._save_csv(rows, cols, 'primary_cutouts', 'total_by_location')

        rows, cols = self._execute_query(f"""
                select substr(datetime, 1, 4) as year, count(*) as count
                from {self.cutout_table}
                where json_extract(cutout_props, '$.is_primary') = true
                group by year
                order by year desc;
                """)
        total_by_year = self._create_table(rows, cols,
                                           "Total Primary Cutouts by Year")
        self._save_csv(rows, cols, 'primary_cutouts', 'total_by_year')

        tables = [x.get_string() for x in
                  [total_count_table, total_by_common_name,
                   total_by_category, total_by_location,
                   total_by_year]]
        self._append_general_dist_txt(tables, 'primary cutouts')

        return


def main(cfg: DictConfig) -> None:
    """Main function to execute report generation and sending it to slack."""
    report = Report(cfg)

    report.copy_relevant_files()
    report.generate_actionable_table()
    log.info(f"Actionable table generated")
    message, files = report.generate_summary_message()
    report.send_slack_notification(message, files)
    log.info(f"Summary info message sent")

    message, files = report.generate_actionable_message()
    report.send_slack_notification(message, files)
    log.info(f"Actionable info message sent")

    dbquery = DBQuery(cfg)
    dbquery.fullsized_data()
    log.info(f"DB - developed image stats saved")
    dbquery.cutout_data()
    log.info(f"DB - cutouts stats saved")
    dbquery.primary_cutout_data()
    log.info(f"DB - primary cutouts stats saved")

    # message block + files to send for general distribution stats
    message_blocks = [
        {
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': f"*SemiF - General Distribution Stats* - "
                        f"{datetime.now().strftime('%m/%d/%Y')}"
            }
        }
    ]
    report.send_slack_notification(message_blocks, [dbquery.txt_db_report])
