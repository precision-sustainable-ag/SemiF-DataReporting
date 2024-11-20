import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from pathlib import Path
import logging

# Set up logging
log = logging.getLogger(__name__)

class CutoutPlotGenerator:
    """
    A class to generate plots from the cutout data.
    
    Args:
        cfg (DictConfig): Configuration object with paths and plot settings.
    """
    
    def __init__(self, cfg: DictConfig):
        log.debug("Initializing CutoutPlotGenerator...")

        self.cutout_tables_dir = Path(cfg.paths.reports_dir, "tables", "cutouts")
        log.debug(f"Cutout tables directory: {self.cutout_tables_dir}")

        # Find the latest CSV file by searching for the filenames without timestamp
        self.common_name_df = self._load_latest_csv(self.cutout_tables_dir, "count_by_common_name")
        log.debug("Loaded common name cutouts table.")
        
        self.location_common_name_df = self._load_latest_csv(self.cutout_tables_dir, "count_by_location_and_common_name")
        log.debug("Loaded location and common name cutouts table.")
        
        self.species_area_class_df = self._load_latest_csv(self.cutout_tables_dir, "count_by_species_and_area_class")
        log.debug("Loaded species and area class cutouts table.")

        self.species_is_primary_df = self._load_latest_csv(self.cutout_tables_dir, "count_by_species_and_is_primary")
        log.debug("Loaded species and is_primary cutouts table.")

        self.species_extends_border_df = self._load_latest_csv(self.cutout_tables_dir, "count_by_species_and_extends_border")
        log.debug("Loaded species and extends border cutouts table.")

        self.species_green_sum_class_df = self._load_latest_csv(self.cutout_tables_dir, "count_by_species_and_green_sum_class")
        log.debug("Loaded species and green sum cutouts table.")

        
        # Create plot directory if not exists
        self.plot_dir = Path(cfg.paths.reports_dir, "plots", "cutouts")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Plot directory created: {self.plot_dir}")
    
    def _load_latest_csv(self, directory: Path, filename_stem: str) -> pd.DataFrame:
        """Helper method to load the latest CSV file based on the filename stem (ignores timestamp)."""
        log.debug(f"Searching for CSV files in {directory} with stem {filename_stem}...")
        # Use glob to find files that start with the given filename stem
        csv_files = list(directory.glob(f"{filename_stem}_*.csv"))
        if not csv_files:
            log.error(f"No CSV files found for {filename_stem} in {directory}")
            raise FileNotFoundError(f"No CSV files found for {filename_stem} in {directory}")
        
        # Load the latest file based on the timestamp in the filename
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        log.debug(f"Latest file selected: {latest_file}")
        return pd.read_csv(latest_file)
    
    def plot_common_name(self):
        """Plot the count of cutouts by common name (species)."""
        log.debug("Generating plot for cutouts by common name...")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.common_name_df.sort_values(by="count", ascending=False), 
                    x="count", y="common_name", palette="viridis", legend=False)
        plt.title("Cutouts by Species (Common Name)")
        plt.xlabel("Count")
        plt.ylabel("Species")
        plt.tight_layout()
        plot_path = self.plot_dir / "cutouts_by_species.png"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"Plot saved to {plot_path}")
    
    def plot_location_and_common_name(self):
        """Plot the count of cutouts by species and location."""
        log.debug("Generating plot for cutouts by species and location...")
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.location_common_name_df, 
                     x="common_name", hue="location", weights="count", 
                     multiple="stack", palette="Set2", shrink=0.8)
        plt.xticks(rotation=90)
        plt.title("Cutouts by Species and Location")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.tight_layout()
        plot_path = self.plot_dir / "cutouts_by_species_and_location.png"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"Plot saved to {plot_path}")
    
    def plot_species_and_area_class(self):
        """Plot the count of cutouts by species and area class."""
        log.debug("Generating plot for cutouts by species and area class...")
        plt.figure(figsize=(12, 8))
        sns.barplot(data=self.species_area_class_df, 
                    x="common_name", y="count", hue="area_class", palette="coolwarm")
        plt.xticks(rotation=90)
        plt.title("Cutouts by Species and Area Class")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.tight_layout()
        plot_path = self.plot_dir / "cutouts_by_species_and_area_class.png"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"Plot saved to {plot_path}")

    def plot_species_and_green_sum_class(self):
        """Plot the count of cutouts by species and green_sum class."""
        log.debug("Generating plot for cutouts by species and green_sum class...")
        plt.figure(figsize=(12, 8))
        sns.barplot(data=self.species_green_sum_class_df, 
                    x="common_name", y="count", hue="green_sum_class", palette="Greens")
        plt.xticks(rotation=90)
        plt.title("Cutouts by Species and green_sum Class")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.tight_layout()
        plot_path = self.plot_dir / "cutouts_by_species_and_green_sum_class.png"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"Plot saved to {plot_path}")


    def plot_species_and_is_primary(self):
        """Plot the count of cutouts by species, split by is_primary (True/False)."""
        log.info("Generating plot for cutouts by species and is_primary (True/False)...")
        plt.figure(figsize=(12, 8))
        sns.barplot(data=self.species_is_primary_df, 
                    x="common_name", y="count", hue="is_primary", palette="Set1")
        plt.xticks(rotation=90)
        plt.title("Cutouts by Species and Is Primary (True/False)")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.tight_layout()
        plot_path = self.plot_dir / "cutouts_by_species_and_is_primary.png"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"Plot saved to {plot_path}")

    def plot_species_and_extends_border(self):
        """Plot the count of cutouts by species, split by extends_border (True/False)."""
        log.info("Generating plot for cutouts by species and extends_border (True/False)...")
        plt.figure(figsize=(12, 8))
        sns.barplot(data=self.species_extends_border_df, 
                    x="common_name", y="count", hue="extends_border", palette="Accent")
        plt.xticks(rotation=90)
        plt.title("Cutouts by Species and Extends Border (True/False)")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.tight_layout()
        plot_path = self.plot_dir / "cutouts_by_species_and_extends_border.png"
        plt.savefig(plot_path)
        plt.close()
        log.info(f"Plot saved to {plot_path}")

def main(cfg: DictConfig):
    log.info("Starting the plot generation process...")
    plot_generator = CutoutPlotGenerator(cfg)
    
    # Generate and save plots
    plot_generator.plot_common_name()
    plot_generator.plot_location_and_common_name()
    plot_generator.plot_species_and_area_class()
    plot_generator.plot_species_and_is_primary()
    plot_generator.plot_species_and_extends_border()
    plot_generator.plot_species_and_green_sum_class()
    log.info("Plot generation process completed.")
