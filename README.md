# SuctionSailLab

## CFD and Machine Learning-Based Aerodynamic Prediction of Suction Sails

SuctionSailLab is a computational tool developed to predict the aerodynamic performance of a two-dimensional suction sail section for wind-assisted ship propulsion.

The application combines machine learning models trained using Computational Fluid Dynamics (CFD) simulation results to predict lift and drag coefficients and generate aerodynamic polars without running new CFD simulations for each prediction.

## Project Objectives

- Investigate the aerodynamic behaviour of a suction sail section.
- Obtain lift and drag coefficients using 2D CFD simulations.
- Build a structured database of CFD results.
- Train machine learning models to predict aerodynamic performance.
- Validate the predictions against reference results.
- Provide an interactive dashboard for exploring aerodynamic parameters and polars.

## Repository Structure

- `Dashboard/`: Streamlit application and Python prediction interface.
- `Database/`: CFD dataset used for model development and validation.
- `MachineLearningModel/`: Final trained MATLAB models and prediction function.
- `MachineLearningAdditional/`: Additional machine learning files, Regression Learner sessions and testing scripts.
- `dashboard_matlab/`: Python virtual environment containing the required Python packages (if included in the repository).
- `Start_Dashboard.bat`: Automatic launcher used to start the application.

## Input Parameters

The dashboard allows the user to specify:

- **Lambda (B/L):** geometric thickness-to-chord ratio.
- **Reynolds number (Re):** calculated from flow velocity and reference chord.
- **Angle of attack (AoA):** angular range used to generate aerodynamic polars.
- **Flap position (x/c):** dimensionless flap location.
- **Suction coefficient (Cq):** parameter representing the applied suction.

## Predicted Outputs

- Lift coefficient, CL.
- Drag coefficient, CD.
- Lift and drag aerodynamic polars as a function of angle of attack.
- A table containing the predicted values.

## Technologies

- Python
- Streamlit
- NumPy
- pandas
- Matplotlib
- MATLAB Engine for Python
- MATLAB Regression Learner
- Computational Fluid Dynamics (CFD)

## Installation and Requirements

Before running SuctionSailLab, the following software must be available on the computer:

### 1. Python

A compatible Python installation is required to run the dashboard and its Python dependencies.

### 2. MATLAB

MATLAB must be installed because the prediction interface uses MATLAB Engine for Python to execute the trained machine learning models.

The MATLAB Engine installation must be compatible with the installed MATLAB release and Python version.

### 3. Project files and Python environment

Download or clone the complete SuctionSailLab repository, keeping the original folder structure.

The project requires a Python virtual environment named `dashboard_matlab`, containing the Python packages needed by the application, including Streamlit, NumPy, pandas, Matplotlib and MATLAB Engine for Python.

The trained MATLAB models and prediction function must also be available in the `MachineLearningModel/` directory.

**Important:** Python virtual environments are generally not portable between computers. If the `dashboard_matlab` environment is not available or does not work on a new computer, it must be recreated using a compatible Python version and the required packages must be installed. MATLAB Engine for Python must be installed for the MATLAB release available on that computer.

## Running the Dashboard

The project includes an automatic launcher:

**`Start_Dashboard.bat`**

This file simplifies the execution of the application. Once the software requirements and Python environment have been configured, the user does not need to open a terminal, activate the Python environment manually, type commands or start MATLAB manually.

To launch the application:

1. Open the main `SuctionSailLab` project folder.
2. Double-click `Start_Dashboard.bat`.
3. The launcher automatically activates the Python environment and starts the Streamlit application.
4. The dashboard opens in the default web browser.

No manual command-line interaction is required.

The application uses MATLAB Engine in the background to access the trained machine learning models and calculate the aerodynamic predictions.

## Data and Validation

The repository includes the CFD database and additional files used for machine learning development and validation. The testing scripts can be used to compare predictions with reference results.

## Limitations

Predictions should be interpreted within the parameter ranges and modelling assumptions represented by the training data. Extrapolation beyond the training domain may not be reliable.

## Author and Purpose

Developed as part of a Master's Thesis (TFM) focused on the aerodynamic modelling of suction sails for wind-assisted ship propulsion.
