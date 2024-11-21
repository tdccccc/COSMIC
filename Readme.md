# COSMIC

COSMIC (Cluster Optical Search using Machine Intelligence in Catalogs) is a galaxy cluster finding algorithm that utilizes machine learning techniques.

## Toy Model

This project selects a 1 square degree region from the SDSS for testing the code.

### Directory Structure

- `data/`: Contains test data and background galaxy data.
- `model/`: Contains the XGBoost model for BCG classification and the ResNet model for richness estimation.
- `output/`: Stores program output data, including candidate BCGs in `BCG_cand.fits`, detected clusters in `cluster.fits`, and other necessary data.
- `source/`: Contains the source code.

## Dependencies

The project requires the following Python packages:

- `astropy==4.3.1`
- `functions==0.7.0`
- `h5py==3.7.0`
- `matplotlib==3.5.3`
- `numpy==1.21.6`
- `pandas==1.2.0`
- `scikit-learn==0.24.2`
- `scipy==1.6.2`
- `torch==1.7.1+cu110`
- `torchvision==0.8.2+cu110`
- `xgboost==1.4.2`

> Note: Ensure you are using a Python environment compatible with these package versions. You can install these dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
## Running the Program

To execute the program, run:

```bash
cd source
python3 run.py
```
