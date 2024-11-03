# COSMIC

COSMIC (Cluster Optical Search using Machine Intelligence in Catalogs) is a galaxy cluster finding algorithm that utilizes machine learning techniques.

## Toy Model

This project selects a 1 square degree region from the SDSS for testing the code.

### Directory Structure

- `data/`: Contains test data and background galaxy data.
- `model/`: Contains the XGBoost model for BCG classification and the ResNet model for richness estimation.
- `output/`: Stores program output data, including candidate BCGs in `BCG_cand.fits`, detected clusters in `cluster.fits`, and other necessary data.
- `source/`: Contains the source code.

## Running the Program

To execute the program, run:

```bash
python source/run.py
