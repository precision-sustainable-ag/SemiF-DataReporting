from pymongo import MongoClient
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
        self.collection = self.db[cfg.mongodb.cutout_collection]
        self.cutout_table_dir = Path(cfg.paths.reports_dir,"tables", "cutouts")
        self.cutout_table_dir.mkdir(parents=True, exist_ok=True)

        log.debug(f"Connected to MongoDB database: {cfg.mongodb.db} and collection: {cfg.mongodb.cutout_collection}")
        log.info(f"Created output directory for cutout tables (if not already existing): {self.cutout_table_dir}")

    def _save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Helper method to save DataFrame to a CSV file.
        """
        timestamp = datetime.now().strftime('%Y%m%d')
        filepath = self.cutout_table_dir / f"{filename}_{timestamp}.csv"
        df.to_csv(filepath, index=False)
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
        log.debug(f"Aggregation completed successfully with {len(data)} records aggregated.")
        
        # Convert to DataFrame and assign column names
        df = pd.DataFrame(data, columns=columns)
        log.debug(f"DataFrame created with columns: {columns} and {len(df)} rows.")
        return df
    
    
    def table_by_species_and_extends_border(self) -> pd.DataFrame:
        """
        Generates a report for the number of cutouts by species, split by the cutout_props.is_primary field (True or False).
        """
        log.debug("Generating report by species and is_primary...")

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "common_name": "$category.common_name",
                        "extends_border": "$cutout_props.extends_border"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.common_name": 1, "_id.extends_border": 1}}  # Sort by species and is_primary
        ]

        # Aggregate data and prepare DataFrame
        columns = ["common_name", "extends_border", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        saved_filepath = self._save_to_csv(df, "count_by_species_and_extends_border")
        log.info(f"Report by species and extends_border saved to: {saved_filepath}")

        return df
    
    def table_by_species_and_is_primary(self) -> pd.DataFrame:
        """
        Generates a report for the number of cutouts by species, split by the cutout_props.is_primary field (True or False).
        """
        log.debug("Generating report by species and is_primary...")

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "common_name": "$category.common_name",
                        "is_primary": "$cutout_props.is_primary"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.common_name": 1, "_id.is_primary": 1}}  # Sort by species and is_primary
        ]

        # Aggregate data and prepare DataFrame
        columns = ["common_name", "is_primary", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        saved_filepath = self._save_to_csv(df, "count_by_species_and_is_primary")
        log.info(f"Report by species and is_primary saved to: {saved_filepath}")

        return df
    
    def table_by_species_and_green_sum_class(self) -> pd.DataFrame:
        """
        Generates a report for the number of cutouts by species for each area class.
        The area classes are in logarithmic ranges (1, 10, 100, 1000, etc.),
        and the report is saved to a CSV file.
        """
        log.debug("Generating report by species and area class...")

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$project": {
                    "common_name": "$category.common_name",
                    "green_sum": "$cutout_props.green_sum"
                }
            },
            {
                "$addFields": {
                    "green_sum_class": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lt": ["$green_sum", 10]}, "then": "1-9"},
                                {"case": {"$lt": ["$green_sum", 100]}, "then": "10-99"},
                                {"case": {"$lt": ["$green_sum", 1000]}, "then": "100-999"},
                                {"case": {"$lt": ["$green_sum", 10000]}, "then": "1000-9999"},
                                {"case": {"$lt": ["$green_sum", 100000]}, "then": "10000-99999"},
                            ],
                            "default": "100000+"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "common_name": "$common_name",
                        "green_sum_class": "$green_sum_class"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.common_name": 1, "_id.green_sum_class": 1}}  # Sort by species and area class
        ]

        # Aggregate data and prepare DataFrame
        columns = ["common_name", "green_sum_class", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        saved_filepath = self._save_to_csv(df, "count_by_species_and_green_sum_class")
        log.info(f"Report by species and green_sum class saved to: {saved_filepath}")

        return df
    
    def table_by_species_and_area_class(self) -> pd.DataFrame:
        """
        Generates a report for the number of cutouts by species for each area class.
        The area classes are in logarithmic ranges (1, 10, 100, 1000, etc.),
        and the report is saved to a CSV file.
        """
        log.debug("Generating report by species and area class...")

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$project": {
                    "common_name": "$category.common_name",
                    "area": "$cutout_props.area"
                }
            },
            {
                "$addFields": {
                    "area_class": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lt": ["$area", 10]}, "then": "1-9"},
                                {"case": {"$lt": ["$area", 100]}, "then": "10-99"},
                                {"case": {"$lt": ["$area", 1000]}, "then": "100-999"},
                                {"case": {"$lt": ["$area", 10000]}, "then": "1000-9999"},
                                {"case": {"$lt": ["$area", 100000]}, "then": "10000-99999"},
                            ],
                            "default": "100000+"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "common_name": "$common_name",
                        "area_class": "$area_class"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.common_name": 1, "_id.area_class": 1}}  # Sort by species and area class
        ]

        # Aggregate data and prepare DataFrame
        columns = ["common_name", "area_class", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        saved_filepath = self._save_to_csv(df, "count_by_species_and_area_class")
        log.info(f"Report by species and area class saved to: {saved_filepath}")

        return df


    
    def table_by_location_and_common_name(self) -> pd.DataFrame:
        """
        Generates a report based on species and location (extracted from cutout_id),
        and saves the report to a CSV file.
        """
        
        log.debug("Generating report by location and common name...")

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "location": {  # Extract location (state abbreviation) from cutout_id
                            "$substrCP": ["$cutout_id", 0, 2]  # Assume location is the first two chars
                        },
                        "common_name": "$category.common_name"
                    },
                    "count": {"$sum": 1}  # Count documents per group
                }
            },
            {"$sort": {"count": -1}}  # Sort by document count in descending order
        ]

        # Aggregate data and prepare DataFrame
        columns = ["location", "common_name", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        saved_filepath = self._save_to_csv(df, "count_by_location_and_common_name")
        log.info(f"Report by location and common name saved to: {saved_filepath}")

        return df
    
    def table_by_common_name(self) -> pd.DataFrame:
        """
        Generates a report based only on species (common name) and count,
        without location information, and saves the report to a CSV file.
        """
        log.debug("Generating report by common name...")

        # MongoDB aggregation pipeline
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "common_name": "$category.common_name"
                    },
                    "count": {"$sum": 1}  # Count documents per group
                }
            },
            {"$sort": {"count": -1}}  # Sort by document count in descending order
        ]

        # Aggregate data and prepare DataFrame
        columns = ["common_name", "count"]
        df = self._aggregate_data(pipeline, columns)

        # Save to CSV
        saved_filepath = self._save_to_csv(df, "count_by_common_name")
        log.info(f"Report by common name saved to: {saved_filepath}")

        return df


def main(cfg: DictConfig) -> None:
    log.info("Starting the MongoDB report generation process...")
    
    # Initialize the MongoTableGenerator with the provided configuration
    table_generator = MongoTableGenerator(cfg)
    
    # Generate and save reports
    log.debug("Generating and saving reports...")
    table_generator.table_by_location_and_common_name()
    table_generator.table_by_common_name()
    table_generator.table_by_species_and_area_class()
    table_generator.table_by_species_and_green_sum_class()
    table_generator.table_by_species_and_is_primary()
    table_generator.table_by_species_and_extends_border()
    
    log.debug("Report generation process completed successfully.")
