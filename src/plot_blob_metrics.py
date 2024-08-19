#!/usr/bin/env python3
import logging
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import os
from utils.utils import read_yaml
import shutil
import re
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as DT

log = logging.getLogger(__name__)




class PlotBlobMetrics:
    """
    Plot blob metrics from the exported blob files.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the BlobMetricsCalculator with configuration data.
        args:
            cfg: The configuration data

        Arguments:
        output_dir: The output directory to save the blob list txt file.
        report_dir: The report directory to save the mismatch statistics.
        """
        self.data_dir = cfg.paths.data_dir
        self.report_dir = cfg.paths.report
        self.matching_folders = cfg.matching_folders
        self.valid_pattern = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'
        log.info("Initialized Exporting BlobMetrics with configuration data.")

    def basic_bar_plot(self, df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, save_path: str) -> None:

        # Define state palette with new labels
        state_palette = {
            "MD": "#4C72B0",
            "NC": "#55A868",
            "TX": "#C44E52",
        }

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bar_plot = sns.barplot(
                data=df,
                x=x,
                y=y,
                hue="State",
                palette=state_palette,
                hue_order=state_palette.keys(),
                ax=ax,
            )
            
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.legend(title="States")
            
            # Add labels to each bar
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type='edge', padding=3, fontsize=7)

            fig.tight_layout()
            save_path = f"{self.data_dir}/{save_path}"
            fig.savefig(save_path, dpi=300)
    
    def basic_line_plot(self, df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
            
            # Define state palette with new labels
            state_palette = {
                "MD": "#4C72B0",
                "NC": "#55A868",
                "TX": "#C44E52",
            }
    
            with plt.style.context("ggplot"):
                fig, ax = plt.subplots(figsize=(16, 8))
                
                line_plot = sns.barplot(
                    data=df,
                    x=y,
                    y=x,
                    hue="State",
                    palette=state_palette,
                    hue_order=state_palette.keys(),
                    ax=ax,
                )
                
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax.set_ylabel(xlabel)
                ax.set_xlabel(ylabel)
                ax.set_title(title)
                ax.legend(title="States")

                # Add labels to each bar
                for bar_container in line_plot.containers:
                    ax.bar_label(bar_container, label_type='edge', padding=3, fontsize=7)
                sns.despine(left=True, bottom=True)
                fig.tight_layout()
                save_path = f"{self.data_dir}/{save_path}"
                fig.savefig(save_path, dpi=300)

    
    def plot_average_delevloped_stats(self, plot_data: pd.DataFrame) -> None:
        """
        Generate a bar plot showing the distribution of images by month, location, "season".
        """
        log.info("Generating bar plot for average number of images by season.")

        season_data=plot_data.groupby(['State', 'Season'])['ImageCount'].mean().reset_index(name='AverageImageCount')

        unique_season_count = (
            season_data.groupby(["Season","State"])['AverageImageCount']
            .sum()
            .reset_index(name="season_count")
        )
        
        
        self.basic_bar_plot(unique_season_count, "Season", "season_count", " Samples by Season", "Season", "Number of JPG Images", "average_season_count.png")

        log.info("Generating bar plot for average number of images by month.")

        month_data=plot_data.groupby(['State', 'Month'])['ImageCount'].mean().reset_index(name='AverageImageCount')

        unique_month_count = (
            month_data.groupby(["Month","State"])['AverageImageCount']
            .sum()
            .reset_index(name="month_count")
        )

        self.basic_line_plot(unique_month_count, "Month", "month_count", " Samples by Month", "Month", "Number of JPG Images", "average_month_count.png")

        log.info("Species distribution for current season plot saved.")

    def plot_average_batch_counts(self, plot_data:pd.DataFrame)->None:
        """
        Generate a bar plot showing the distribution of batches by month.
        """
        log.info("Generating bar plot for average number of batches by month.")
        
        
        self.basic_line_plot(plot_data, "Month", "BatchCount", " Batches by Month", "Month", "Number of Batches", "average_batch_count.png")

    def plot_average_cut_out_counts(self, plot_data:pd.DataFrame)->None:
        """
        Generate a bar plot showing the distribution of cut outs by month.
        """
        log.info("Generating bar plot for average number of cut outs by month.")
        
        
        self.basic_line_plot(plot_data, "Month", "AverageMonthlyCutoutCount", " Cut Outs by Month", "Month", "Number of Cut Outs", "average_cut_out_count.png")

    def load_data(self, data_file: Path) -> pd.DataFrame:
        """Loads the data from the given path.
        Args:
            path: The path to the data file.
        Returns:
            The DataFrame containing the data.
            """
        file_path=Path(self.data_dir, data_file)
        # Load the text file
        df = pd.read_csv(file_path)

        #Extract the batches and create a DataFrame
        log.info(f"Created DataFrame with {len(df)} rows.")

        return df
    

def main(cfg: DictConfig) -> None:
    """Main function to execute the BlobMetricPlotter."""


    plotter = PlotBlobMetrics(cfg)
 

    #Loading saved processed csv data 
    #Please delete the unwanted csv files lines
    mismatch_statistics_developed=plotter.load_data('blob_containers/mismatch_statistics_developed.csv')
    unprocessed_batches_developed=plotter.load_data('blob_containers/unprocessed_batches_developed.csv')

    mismatch_statistics_cutout=plotter.load_data('blob_containers/mismatch_statistics_cutout.csv')
    unprocessed_batches_cutout=plotter.load_data('blob_containers/unprocessed_batches_cutout.csv')

    dataset_statistics_upload=plotter.load_data('blob_containers/dataset_statistics_upload.csv')
    uncolorized_batches=plotter.load_data('blob_containers/uncolorized_batches.csv')

    upload_recent_df=plotter.load_data('blob_containers/upload_recent_df.csv')
    colorized_recent_df=plotter.load_data('blob_containers/colorized_recent_df.csv')

    season_image_counts=plotter.load_data('blob_containers/season_image_counts.csv')
    average_batch_counts=plotter.load_data('blob_containers/average_batch_counts.csv')

    average_cropout_counts=plotter.load_data('blob_containers/average_cutout_counts.csv')

    #plot average season and month image counts
    plotter.plot_average_delevloped_stats(season_image_counts)
    #plot average batch counts
    plotter.plot_average_batch_counts(average_batch_counts)
    #plot average cut out counts
    plotter.plot_average_cut_out_counts(average_cropout_counts)



    
    log.info(f"{cfg.task} completed.")