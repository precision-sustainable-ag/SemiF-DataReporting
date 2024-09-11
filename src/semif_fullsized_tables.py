from pymongo import MongoClient
from pymongo import ASCENDING
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

# Set up logging
log = logging.getLogger(__name__)

class MongoTableGenerator:
    """
    A class to generate a MongoDB report based on species and location from a collection.

    Args:
        cfg (DictConfig): Configuration object for MongoDB connection details.
    """
    
    def __init__(self, cfg: DictConfig):
        # MongoDB connection setup
        log.debug("Initializing MongoDB client and connecting to database...")
        self.client = MongoClient(host=cfg.mongodb.host, port=cfg.mongodb.port)
        self.db = self.client[cfg.mongodb.db]
        self.collection = self.db[cfg.mongodb.fullsized_collection]
        self.table_dir = Path(cfg.paths.tables_dir, "fullsized")
        self.table_dir.mkdir(parents=True, exist_ok=True)

        log.debug(f"Connected to MongoDB database: {cfg.mongodb.db} and collection: {cfg.mongodb.fullsized_collection}")
        log.debug(f"Output directory for tables: {self.table_dir}")

    def _save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Helper method to save DataFrame to a CSV file.
        """
        timestamp = datetime.now().strftime('%Y%m%d')
        filepath = self.table_dir / f"{filename}_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        log.debug(f"Report saved to CSV: {filepath}")
        return filepath
    
    def _aggregate_data(self, pipeline: list, columns: list) -> pd.DataFrame:
        """
        Helper method to aggregate data from MongoDB using the given pipeline.
        """
        log.debug("Executing MongoDB aggregation pipeline...")
        results = self.collection.aggregate(pipeline)
        data = []
        for result in results:
            # Flatten the '_id' field into individual fields
            flattened_result = {**result["_id"], "count": result["count"]}
            data.append(flattened_result)
        log.debug(f"Aggregation completed with {len(data)} records")

        # Convert to DataFrame and assign column names
        df = pd.DataFrame(data, columns=columns)
        return df
    
    def table_by_common_name(self) -> pd.DataFrame:
        """
        Generates a report based on species (common_name) only, and saves the report to a CSV file.
        """
        log.debug("Generating report by common name...")

        # MongoDB aggregation pipeline
        pipeline = [
            # Step 1: Unwind the categories array to access each category's common_name field
            {
                "$unwind": "$categories"
            },
            # Step 2: Project the common_name field
            {
                "$project": {
                    "common_name": "$categories.common_name"  # Access the common_name field in categories
                }
            },
            # Step 3: Group by `common_name` and count the number of documents per group
            {
                "$group": {
                    "_id": {
                        "common_name": "$common_name"  # Group by the common name of the species
                    },
                    "count": { "$sum": 1 }  # Count the number of documents for each species
                }
            },
            # Sort by common name for readability (ascending order)
            {
                "$sort": { "_id.common_name": ASCENDING }
            }
        ]

        # Aggregate data and prepare DataFrame
        columns = ["common_name", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        self._save_to_csv(df.sort_values(by="count", ascending=False), "count_by_common_name")

        return df

    def table_by_location_and_common_name(self) -> pd.DataFrame:
        """
        Generates a report based on species and location (extracted from image_id),
        and saves the report to a CSV file.
        """
        log.debug("Generating report by location and common name...")

        # MongoDB aggregation pipeline
        pipeline = [
            # Step 1: Unwind the categories array to access each category's common_name field
            {
                "$unwind": "$categories"
            },
            # Step 2: Project the location and the common_name fields
            {
                "$project": {
                    "location": { "$substr": ["$batch_id", 0, 2] },  # Extract the first 2 characters from batch_id
                    "common_name": "$categories.common_name"  # Access the common_name field in categories
                }
            },
            # Step 3: Group by both `location` and `common_name`, and count the number of documents per group
            {
                "$group": {
                    "_id": {
                        "location": "$location",  # Group by the location field
                        "common_name": "$common_name"  # Group by the common name of the species
                    },
                    "count": { "$sum": 1 }  # Count the number of documents in each group
                }
            },
            # Optional: Sort by location and then by common name for readability (ascending order)
            {
                "$sort": { "_id.location": ASCENDING, "_id.common_name": ASCENDING }
            }
        ]

        # Aggregate data and prepare DataFrame
        columns = ["location", "common_name", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        self._save_to_csv(df.sort_values(by="count", ascending=False), "count_by_location_and_common_name")

        return df


def main(cfg: DictConfig) -> None:
    log.info("Starting the MongoDB report generation process...")
    
    # Initialize the MongoReportGenerator with the provided configuration
    table_generator = MongoTableGenerator(cfg)
    
    # Generate and save reports
    log.info("Generating and saving reports...")
    table_generator.table_by_location_and_common_name()
    table_generator.table_by_common_name()
    
    log.info("Report generation process completed successfully")
