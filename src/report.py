import logging
import shutil
import os
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
        
    def _copy_relevant_files(self):
        # get relevant blobcontainers files
        # pass in the filename for the latest csvs
        # Copies file with metadata (creation and modification times)
        blob_container_folder = os.path.join(self.source_location, 'blob_containers')
        
        shutil.copy2(os.path.join(blob_container_folder, 'semif-HighLevelStats.txt'), self.report_folder)
    
    def compose_slack_message(self):
        self._copy_relevant_files()
        
        with open(os.path.join(self.report_folder, 'semif-HighLevelStats.txt'), 'r') as f:
            self.message_blocks.append({
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': ''.join(f.readlines())
                }
            })

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
            # response = client.chat_postMessage(**message)
            # print("response 1: ", response)
            file_response = client.files_upload_v2(
                # channel=response['channel'],
                channel='C083GS3GD5Z',
                file=os.path.join(self.source_location, 'blob_containers', 'semif-HighLevelStats.txt'),
                # initial_comment="Here's the attached file",
                # thread_ts=response['ts'],
                thread_ts='1734350163.621779'
            )
            print("response 2: ", file_response)
            # if response["ok"]:
            #     print("Message sent successfully")
        except SlackApiError as e:
            print(f"Error: {e}")

def main(cfg: DictConfig) -> None:
    """Main function to execute report generation and sending it to slack."""
    report = Report(cfg)
    log.info('generating report')
    report.compose_slack_message()
    log.info('sending slack notification')
    report.send_slack_notification()
