import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
import logging
from utils.utils import get_most_recent_csv

log = logging.getLogger(__name__)


class SpeciesDataProcessor:
    planttype_palette = {
        "weeds": "#55A868",
        "cover crops": "#4C72B0",
        "cash": "#C44E52",
    }

    def __init__(self, csv_path: Path, output_dir: str) -> None:
        """Initialize the data processor with the CSV path and output directory."""
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.categories_of_interest = self.define_categories()
        self.df = self.load_and_preprocess_data()

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess data from the CSV file."""
        log.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df.columns = [col.lower() if col not in ['batch', 'FormatType', 'ImageCount'] else col for col in df.columns]
        log.info("Data loaded successfully. Preprocessing data.")
        df = self.preprocess_data(df)
        log.info("Data preprocessed successfully.")
        return df

    @staticmethod
    def define_categories() -> dict:
        """Define categories of interest for species."""
        log.info("Defining categories of interest.")
        categories = {
            'weeds': ['Palmer amaranth', 'Common ragweed', 'Sicklepod', 'Cocklebur', 'Large crabgrass', 'Goosegrass', 
                      'Broadleaf signalgrass', 'Purple nutsedge', 'Waterhemp', 'Barnyardgrass', 'Jungle rice', 'Texas millet', 
                      'Kochia', 'Common sunflower', 'Ragweed parthenium', 'Johnsongrass', 'Smooth pigweed', 'Common lambsquarters', 
                      'Fall panicum', 'Jimson weed', 'Velvetleaf', 'Yellow foxtail', 'Giant foxtail', 'Horseweed', 'Sprawling signalgrass', 
                      'Prickly sida', 'Pitted morning-glory', 'Spiny amaranth', 'desert horsepurslane'],
            'cover crops': ['Hairy vetch', 'Winter pea', 'Crimson clover', 'Red clover', 'Mustards', 'cultivated radish', 
                            'Cereal rye', 'Triticale', 'Winter wheat', 'Oats', 'Barley', 'Black oats'],
            'cash': ['upland cotton', 'Soybean', 'Maize']
        }
        return {k: [s.lower() for s in v] for k, v in categories.items()}

    def map_species_to_category(self, species: str) -> str:
        """Map species to their respective category."""
        species = species.split("_")[0].lower()
        for category, species_list in self.categories_of_interest.items():
            if species in species_list:
                return category
        return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for analysis and plotting."""
        log.info("Melting the DataFrame to long format.")
        df_long = df.melt(id_vars=['batch', 'FormatType', 'ImageCount'], var_name='species', value_name='count')
        df_long = df_long[df_long['count'] > 0]
        df_long['species'] = df_long['species'].str.replace('_count', '')
        df_long = df_long[~df_long['species'].str.contains('primary')]
        df_long['category'] = df_long['species'].apply(self.map_species_to_category)

        log.info("Processing primary counts.")
        primary_cols = [col for col in df.columns if col.endswith('_primary_count')]
        df_long_primary = df.melt(id_vars=['batch', 'FormatType', 'ImageCount'], value_vars=primary_cols, 
                                            var_name='species_primary', value_name='primary_count')
        df_long_primary['species_primary'] = df_long_primary['species_primary'].str.replace('_primary_count', '')
        df_long_primary = df_long_primary[df_long_primary['primary_count'] > 0]
        df_long_primary['category'] = df_long_primary['species_primary'].apply(self.map_species_to_category)

        log.info("Merging total and primary counts DataFrames.")
        df_merged = pd.merge(df_long, df_long_primary, left_on=['batch', 'species', 'category'], 
                                  right_on=['batch', 'species_primary', 'category'], how='left')
        df_merged['primary_count'] = df_merged['primary_count'].fillna(0)
        df_filtered = df_merged[df_merged['category'].notnull()]
        df_filtered = df_filtered.rename(columns={'species': 'common_name', 'ImageCount_x': 'ImageCount', 
                                                            'count': 'cutout_count', 'primary_count': 'primary_cutout_count'})
        return df_filtered.sort_values(by="batch")

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to a CSV file."""
        log.info(f"Saving DataFrame to {filename}")
        df.to_csv(self.output_dir / filename, index=False)
        log.info(f"DataFrame saved to {filename}")

    def plot_category_data(self, df: pd.DataFrame, y_column: str, title_prefix: str, filename_prefix: str) -> None:
        """Plot data for each category and save the plots."""
        log.info(f"Plotting data for column: {y_column}")
        categories = df['category'].unique()
        for category in categories:
            log.info(f"Plotting data for category: {category}")
            df_category = df[df['category'] == category]
            g = sns.catplot(data=df_category, kind="bar", x='common_name', y=y_column, height=6, aspect=2, color=self.planttype_palette.get(category))
            rotation = 85 if category == "weeds" else 65
            g.set_xticklabels(rotation=rotation, ha='center')
            
            self.add_labels_to_plot(g)
            plt.title(f'{title_prefix} ({category})', y=1.03)
            plt.tight_layout()
            plot_path = self.output_dir / f"{filename_prefix}_{category}.png"
            plt.savefig(plot_path)
            log.info(f"Plot saved to {plot_path}")

    @staticmethod
    def add_labels_to_plot(plot: sns.axisgrid.FacetGrid) -> None:
        """Add labels to bars in the plot."""
        log.info("Adding labels to the plot.")
        for ax in plot.axes.flatten():
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    def process_and_plot(self) -> None:
        """Process data and generate plots and CSV files."""
        log.info("Processing and plotting data.")
        df_image_count = self.df.groupby(['category', 'common_name'])['ImageCount'].sum().reset_index().sort_values(by=["ImageCount"])
        self.save_csv(df_image_count, "total_images_by_species.csv")
        self.plot_category_data(df_image_count, 'ImageCount', 'Sum of Images by Species', 'total_images_by_species_for')

        df_total_cutouts = self.df.groupby(['category', 'common_name'])['cutout_count'].sum().reset_index().sort_values(by=["cutout_count"])
        self.save_csv(df_total_cutouts, "total_cutouts_by_species.csv")
        self.plot_category_data(df_total_cutouts, 'cutout_count', 'Sum of Total Cutouts (primary and non_primary) by Species', 'total_cutouts_by_species_for')

        df_primary_count = self.df.groupby(['category', 'common_name'])['primary_cutout_count'].sum().reset_index().sort_values(by=["primary_cutout_count"])
        self.save_csv(df_primary_count, "primary_cutouts_by_species.csv")
        self.plot_category_data(df_primary_count, 'primary_cutout_count', 'Sum of Total Primary Cutouts by Species', 'primary_cutouts_by_species_for')
        log.info("Data processing and plotting completed.")

def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.task}")
    folder_path = Path(cfg.paths.data_dir, "species_distribution")
    csv_path = get_most_recent_csv(folder_path)

    output_dir =  Path(cfg.paths.report, "semif_report_2024_07_19")
    output_dir.mkdir(exist_ok=True, parents=True)
    processor = SpeciesDataProcessor(csv_path, output_dir)
    processor.process_and_plot()
    log.info(f"{cfg.task} compeleted.")
