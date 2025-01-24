import sys
from pathlib import Path
import logging
import datetime
# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf  # Do not confuse with dataclass.MISSING
from hydra.utils import get_method

log = logging.getLogger(__name__)
# Get the logger for the Azure SDK
azlogger = logging.getLogger("azure")
# Set the logging level to CRITICAL to turn off regular logging
azlogger.setLevel(logging.WARN)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.pipeline)}")
    
    # Compute the current date
    now_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Replace placeholder with the current date
    for region in cfg.bbot_versions.keys():
        for version in cfg.bbot_versions[region].keys():
            cfg.bbot_versions[region][version] = [
                date if date != "NOW_DATE_PLACEHOLDER" else now_date
                for date in cfg.bbot_versions[region][version]
            ]
    
    for tsk in cfg.pipeline:
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    main()