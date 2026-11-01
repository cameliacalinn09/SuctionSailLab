from pathlib import Path

import matlab
import matlab.engine
import numpy as np
import pandas as pd


# ============================================================
# PROJECT CONSTANTS
# ============================================================

# Air properties
RHO_AIR = 1.225       # Air density [kg/m^3]
MU_AIR = 1.8e-5       # Dynamic viscosity [Pa·s]

# Reference chord
CHORD = 1.0           # Chord [m]


# ============================================================
# TRAINING DOMAIN
# ============================================================

# Geometric ratio lambda = B/L
LAMBDA_MIN = 0.1
LAMBDA_MAX = 1.0

# Reynolds number
RE_MIN = 2.0e6
RE_MAX = 6.0e6
# User-friendly velocity limits rounded to two decimals. These allow a very
# small extrapolation beyond both ends of the nominal training Re range.
VELOCITY_MIN = 29.38
VELOCITY_MAX = 88.17
RE_MIN_ALLOWED = RHO_AIR * VELOCITY_MIN * CHORD / MU_AIR
RE_MAX_ALLOWED = RHO_AIR * VELOCITY_MAX * CHORD / MU_AIR

# Angle of attack
AOA_MIN_TRAINED = 0.0
AOA_MAX_TRAINED = 30.0

# Flap position x/c
X_MIN = 0.5
X_MAX = 0.875

# Suction coefficient
CQ_MIN = 0.0
CQ_MAX = 0.15


# ============================================================
# MATLAB / ML FOLDER
# ============================================================

# Resolve paths relative to the repository, so the project can be moved
# to another folder or cloned on another computer.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATLAB_FOLDER = str(PROJECT_ROOT / "MachineLearningModel")


# ============================================================
# REYNOLDS NUMBER
# ============================================================

def calculate_reynolds(velocity):
    """
    Calculate the Reynolds number from the wind speed.

    Re = rho * V * c / mu

    Parameters
    ----------
    velocity : float
        Wind speed [m/s]

    Returns
    -------
    float
        Reynolds number
    """

    if velocity <= 0:
        raise ValueError(
            "Wind speed must be greater than zero."
        )

    Re = RHO_AIR * velocity * CHORD / MU_AIR

    # Correct negligible floating-point differences at the trained upper bound.
    if abs(Re - RE_MAX) < 0.1:
        Re = RE_MAX

    return Re


# ============================================================
# VELOCITY VALIDATION
# ============================================================

def validate_reynolds(Re):
    """
    Check that the calculated Reynolds number is within
    the ML training domain.

    Parameters
    ----------
    Re : float
        Reynolds number
    """

    # Accept the rounded, user-friendly velocity limits (29.38 and 88.17 m/s),
    # even though they place Re slightly outside the nominal training domain.
    # A tiny numerical tolerance avoids false rejection due to floating-point rounding.
    tolerance = 1.0
    if Re < (RE_MIN_ALLOWED - tolerance) or Re > (RE_MAX_ALLOWED + tolerance):
        raise ValueError(
            f"Reynolds number is outside the allowed prediction range. "
            f"Allowed range for the selected velocity limits: "
            f"{RE_MIN_ALLOWED:.0f} to {RE_MAX_ALLOWED:.0f}. "
            f"Calculated value: {Re:.0f}."
        )


# ============================================================
# ANGLE OF ATTACK VECTOR
# ============================================================

def generate_aoa(aoa_min, aoa_max, aoa_step):
    """
    Generate the AoA vector used for the aerodynamic polar.

    Parameters
    ----------
    aoa_min : float
        Minimum angle of attack [deg]

    aoa_max : float
        Maximum angle of attack [deg]

    aoa_step : float
        AoA interval [deg]

    Returns
    -------
    numpy.ndarray
        AoA vector
    """

    # --------------------------------------------------------
    # Check AoA minimum
    # --------------------------------------------------------

    if aoa_min < AOA_MIN_TRAINED:
        raise ValueError(
            f"AoA minimum must be greater than or equal to "
            f"{AOA_MIN_TRAINED} degrees."
        )

    # --------------------------------------------------------
    # Check AoA maximum
    # --------------------------------------------------------

    if aoa_max > AOA_MAX_TRAINED:
        raise ValueError(
            f"AoA maximum must be lower than or equal to "
            f"{AOA_MAX_TRAINED} degrees."
        )

    # --------------------------------------------------------
    # Check AoA order
    # --------------------------------------------------------

    if aoa_min >= aoa_max:
        raise ValueError(
            "AoA minimum must be smaller than AoA maximum."
        )

    # --------------------------------------------------------
    # Check AoA interval
    # --------------------------------------------------------

    if aoa_step <= 0:
        raise ValueError(
            "AoA interval must be greater than zero."
        )

    # --------------------------------------------------------
    # Generate AoA vector
    # --------------------------------------------------------

    aoa = np.arange(
        aoa_min,
        aoa_max + aoa_step,
        aoa_step
    )

    # Do not exceed requested maximum
    aoa = aoa[aoa <= aoa_max]

    # --------------------------------------------------------
    # Check number of points
    # --------------------------------------------------------

    if len(aoa) < 2:
        raise ValueError(
            "The selected AoA range and interval do not "
            "generate enough points."
        )

    return aoa


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def validate_parameters(
    lambda_value,
    x,
    cq
):
    """
    Validate the geometric and suction parameters against
    the ML training domain.

    Parameters
    ----------
    lambda_value : float
        Geometric ratio B/L

    x : float
        Flap position x/c

    cq : float
        Suction coefficient
    """

    # --------------------------------------------------------
    # Lambda
    # --------------------------------------------------------

    if lambda_value < LAMBDA_MIN or lambda_value > LAMBDA_MAX:
        raise ValueError(
            f"Lambda is outside the ML training domain. "
            f"Valid range: {LAMBDA_MIN} to {LAMBDA_MAX}. "
            f"Selected value: {lambda_value}."
        )

    # --------------------------------------------------------
    # Flap position x/c
    # --------------------------------------------------------

    if x < X_MIN or x > X_MAX:
        raise ValueError(
            f"Flap position x/c is outside the ML training domain. "
            f"Valid range: {X_MIN} to {X_MAX}. "
            f"Selected value: {x}."
        )

    # --------------------------------------------------------
    # Suction coefficient Cq
    # --------------------------------------------------------

    if cq < CQ_MIN or cq > CQ_MAX:
        raise ValueError(
            f"Suction coefficient Cq is outside the ML training "
            f"domain. Valid range: {CQ_MIN} to {CQ_MAX}. "
            f"Selected value: {cq}."
        )


# ============================================================
# START MATLAB
# ============================================================

def start_matlab():
    """
    Start MATLAB and move to the folder containing the
    trained ML models and predictSuctionSail.m.

    Returns
    -------
    matlab.engine.MatlabEngine
        MATLAB engine object
    """

    matlab_folder = Path(MATLAB_FOLDER)

    if not matlab_folder.is_dir():
        raise FileNotFoundError(
            f"MATLAB model folder was not found: {matlab_folder}"
        )

    required_files = (
        "predictSuctionSail.m",
        "trainedModel_CL.mat",
        "trainedModel_CD.mat",
    )
    missing_files = [
        filename for filename in required_files
        if not (matlab_folder / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "The following required MATLAB files are missing from "
            f"{matlab_folder}: {', '.join(missing_files)}"
        )

    eng = matlab.engine.start_matlab()

    eng.cd(
        str(matlab_folder),
        nargout=0
    )

    return eng


# ============================================================
# PREDICT AERODYNAMIC POLAR
# ============================================================

def predict_polar(
    eng,
    lambda_value,
    x,
    cq,
    velocity,
    aoa_min,
    aoa_max,
    aoa_step
):
    """
    Predict the aerodynamic polar of the suction sail.

    Parameters
    ----------
    eng : MATLAB engine
        Active MATLAB engine

    lambda_value : float
        Geometric ratio B/L

    x : float
        Flap position x/c

    cq : float
        Suction coefficient

    velocity : float
        Wind speed [m/s]

    aoa_min : float
        Minimum AoA [deg]

    aoa_max : float
        Maximum AoA [deg]

    aoa_step : float
        AoA interval [deg]

    Returns
    -------
    dict
        Dictionary containing:

        velocity
        Reynolds number
        lambda
        x/c
        Cq
        AoA vector
        CL vector
        CD vector
        maximum CL
        AoA at maximum CL
        minimum CD
        AoA at minimum CD
        results DataFrame
    """

    # ========================================================
    # 1. VALIDATE GEOMETRIC AND SUCTION PARAMETERS
    # ========================================================

    validate_parameters(
        lambda_value,
        x,
        cq
    )

    # ========================================================
    # 2. CALCULATE REYNOLDS NUMBER
    # ========================================================

    Re = calculate_reynolds(velocity)

    # ========================================================
    # 3. VALIDATE REYNOLDS NUMBER
    # ========================================================

    validate_reynolds(Re)

    # ========================================================
    # 4. GENERATE AoA VECTOR
    # ========================================================

    aoa = generate_aoa(
        aoa_min,
        aoa_max,
        aoa_step
    )

    # ========================================================
    # 5. CONVERT AoA VECTOR TO MATLAB FORMAT
    # ========================================================

    aoa_matlab = matlab.double(
        aoa.tolist()
    )

    # ========================================================
    # 6. CALL MATLAB PREDICTION FUNCTION
    # ========================================================

    CL, CD = eng.predictSuctionSail(
        float(lambda_value),
        float(Re),
        aoa_matlab,
        float(x),
        float(cq),
        nargout=2
    )

    # ========================================================
    # 7. CONVERT MATLAB RESULTS TO NUMPY
    # ========================================================

    CL = np.asarray(CL).flatten()
    CD = np.asarray(CD).flatten()

    # ========================================================
    # 8. CHECK RESULT DIMENSIONS
    # ========================================================

    if len(CL) != len(aoa):
        raise ValueError(
            "The number of predicted CL values does not "
            "match the number of AoA points."
        )

    if len(CD) != len(aoa):
        raise ValueError(
            "The number of predicted CD values does not "
            "match the number of AoA points."
        )

    # ========================================================
    # 9. CREATE RESULTS DATAFRAME
    # ========================================================

    results = pd.DataFrame({
        "AoA [deg]": aoa,
        "CL": CL,
        "CD": CD
    })

    # ========================================================
    # 10. MAXIMUM LIFT COEFFICIENT
    # ========================================================

    max_cl_index = np.argmax(CL)

    max_CL = CL[max_cl_index]

    AoA_at_max_CL = aoa[max_cl_index]

    # ========================================================
    # 11. MINIMUM DRAG COEFFICIENT
    # ========================================================

    min_cd_index = np.argmin(CD)

    min_CD = CD[min_cd_index]

    AoA_at_min_CD = aoa[min_cd_index]

    # ========================================================
    # 12. RETURN RESULTS
    # ========================================================

    return {
        "velocity": velocity,
        "Re": Re,
        "lambda": lambda_value,
        "x/c": x,
        "Cq": cq,
        "AoA": aoa,
        "CL": CL,
        "CD": CD,
        "max_CL": max_CL,
        "AoA_at_max_CL": AoA_at_max_CL,
        "min_CD": min_CD,
        "AoA_at_min_CD": AoA_at_min_CD,
        "results": results
    }