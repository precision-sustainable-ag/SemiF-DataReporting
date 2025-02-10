#!/bin/bash

# Assumptions
# conda is installed
# conda env semif_annotation is created
# uses jbshah's repo for generating report

# path to conda
source /home/jbshah/miniconda3/etc/profile.d/conda.sh
conda activate semif_annotation

cd /home/jbshah/SemiF-DataReporting/
cp /mnt/research-projects/s/screberg/longterm_images2/semifield-database/agir.db  ./agir.db
python /home/jbshah/SemiF-DataReporting/main.py

# crontab -l
# 0 9 * * 1 /home/jbshah/SemiF-DataReporting/cronjob.sh