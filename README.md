# SemiF-DataReporting

This repository provides detailed reports on Semifield image data, including
data contents, species distribution, temporal and spatial distribution, missing
data analysis, and status of unprocessed or backlog data.

## [Documentation](https://precision-sustainable-ag.atlassian.net/wiki/spaces/IR2/pages/932806674/SemiField+Data+Reporting)


## Installation
The current codebase is configured to run `SUNNY` server under `jbshah` 
user's codebase. To replicate it under your own user space, change `path` 
under `db` config to point to either NFS file location of the database or 
your copy of the SQLite database file. Using the NFS location as the 
database would result in significantly higher query execution time. 

**Environment installation**
```bash
conda env create --name semif_datareporting --file=environments.yaml
```

**Cronjob**

To run as linux cronjob, edit `cronjob.sh` bash file to point to your 
installation of `SemiF-DataReporting` repo.
