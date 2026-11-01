from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from suction_sail_predictor import (
    LAMBDA_MIN,
    LAMBDA_MAX,
    X_MIN,
    X_MAX,
    CQ_MIN,
    CQ_MAX,
    RE_MIN,
    RE_MAX,
    AOA_MIN_TRAINED,
    AOA_MAX_TRAINED,
    RHO_AIR,
    MU_AIR,
    CHORD,
    start_matlab,
    predict_polar,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Suction Sail Aerodynamic Predictor",
    layout="wide",
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        color: #00064D;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #000000;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        color: #00064D;
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.0rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        background-color: #0261C8;
        color: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .result-title {
        font-size: 0.90rem;
        margin-bottom: 0.35rem;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 700;
    }

    .result-subtitle {
        font-size: 0.80rem;
        margin-top: 0.25rem;
    }

    div.stButton > button {
        width: 100%;
        background-color: #00064D;
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: 600;
        padding: 0.65rem 1rem;
    }

    div.stButton > button:hover {
        background-color: #00064D;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MATLAB ENGINE
# ============================================================

@st.cache_resource
def get_matlab_engine():
    """Start MATLAB once and keep the engine alive."""
    return start_matlab()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def plot_polar(ax, aoa, values, ylabel, title):
    """Plot the MATLAB predictions directly, without interpolation."""

    aoa = np.asarray(aoa, dtype=float)
    values = np.asarray(values, dtype=float)

    # Plot the original MATLAB predictions only: no interpolation or resampling.
    # Markers make every predicted AoA visible, as in the MATLAB figures.
    marker = "o" if "Lift" in title else "s"
    ax.plot(
        aoa,
        values,
        linewidth=1.6,
        color="#0875C9",
        marker=marker,
        markersize=5.0,
        markerfacecolor="none",
        markeredgewidth=1.4,
    )

    ax.set_xlabel("Angle of Attack, AoA [°]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # The horizontal axis covers exactly the AoA requested by the user.
    ax.set_xlim(float(aoa.min()), float(aoa.max()))

    # Show a tick at every AoA requested by the user.
    if len(aoa) > 1:
        aoa_step = np.min(np.diff(aoa))
        ticks = np.arange(
            aoa.min(),
            aoa.max() + aoa_step * 0.5,
            aoa_step
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:g}" for x in ticks])


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Suction Sail Aerodynamic Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    This tool predicts the aerodynamic performance of a suction sail
    using Machine Learning models trained in MATLAB.
    <br>
    Enter the physical and operating parameters below to calculate
    the aerodynamic performance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">Input Parameters</div>',
    unsafe_allow_html=True,
)

input_col_1, input_col_2, input_col_3 = st.columns(
    [1, 1, 1.05],
    gap="medium",
)


# ------------------------------------------------------------
# FIRST COLUMN: FOUR INPUT PARAMETERS
# ------------------------------------------------------------

with input_col_1:

    lambda_value = st.number_input(
        "Lambda",
        min_value=float(LAMBDA_MIN),
        max_value=float(LAMBDA_MAX),
        value=0.50,
        step=0.01,
        format="%.2f",
        help="Geometric ratio B/L.",
    )

    st.caption(f"Range: {LAMBDA_MIN:g} - {LAMBDA_MAX:g}")

    x_value = st.number_input(
        "Flap position x/c",
        min_value=float(X_MIN),
        max_value=float(X_MAX),
        value=0.625,
        step=0.005,
        format="%.3f",
        help="Flap position expressed as x/c.",
    )

    st.caption(f"Range: {X_MIN:g} - {X_MAX:g}")

    cq_value = st.number_input(
        "Suction coefficient Cq",
        min_value=float(CQ_MIN),
        max_value=float(CQ_MAX),
        value=0.050,
        step=0.005,
        format="%.3f",
        help="Suction coefficient used by the ML model.",
    )

    st.caption(f"Range: {CQ_MIN:g} - {CQ_MAX:g}")

    # User-friendly limits rounded to two decimals; both may be slightly
    # outside the nominal Reynolds-number training range.
    velocity_min = 29.38
    velocity_max = 88.17

    velocity = st.number_input(
        "Flow velocity V [m/s]",
        min_value=float(velocity_min),
        max_value=float(velocity_max),
        value=50.0,
        step=0.01,
        format="%.2f",
        help="Wind speed used to calculate Reynolds number. Limits are 29.38–88.17 m/s, so the calculated Reynolds number may be slightly outside the training range.",
    )

    st.caption(
        f"Range: {velocity_min:.2f} - {velocity_max:.2f} m/s"
    )


# ------------------------------------------------------------
# SECOND COLUMN: AoA PARAMETERS AND REYNOLDS NUMBER
# ------------------------------------------------------------

with input_col_2:

    aoa_min = st.number_input(
        "Angle of Attack minimum [°]",
        min_value=float(AOA_MIN_TRAINED),
        max_value=float(AOA_MAX_TRAINED),
        value=0.0,
        step=1.0,
        format="%.1f",
    )

    st.caption(
        f"Range: {AOA_MIN_TRAINED:g}° - {AOA_MAX_TRAINED:g}°"
    )

    aoa_max = st.number_input(
        "Angle of Attack maximum [°]",
        min_value=float(AOA_MIN_TRAINED),
        max_value=float(AOA_MAX_TRAINED),
        value=30.0,
        step=1.0,
        format="%.1f",
    )

    st.caption(
        f"Range: {AOA_MIN_TRAINED:g}° - {AOA_MAX_TRAINED:g}°"
    )

    aoa_step = st.number_input(
        "Angle of Attack interval [°]",
        min_value=0.1,
        max_value=30.0,
        value=5.0,
        step=0.5,
        format="%.1f",
    )

    Re_preview = RHO_AIR * velocity * CHORD / MU_AIR
    # Keep the calculated Reynolds number visible, including the small overshoot at 88.17 m/s.

    reynolds_html = (
        '<div style="'
        'background-color:#0261C8;'
        'color:white;'
        'border-radius:7px;'
        'padding:7px 12px;'
        'margin-top:10px;'
        'width:78%;'
        '">'
        '<div style="font-size:11px; margin-bottom:1px;">'
        'Calculated Reynolds'
        '</div>'
        f'<div style="font-size:20px; font-weight:500; line-height:1.1; margin-bottom:2px;">'
        f'{Re_preview:,.0f}'
        '</div>'
        '<div style="font-size:10px;">'
        'Re = &rho; V L / &mu;'
        '</div>'
        f'<div style="font-size:9px; margin-top:2px;">'
        f'Range: {RHO_AIR * velocity_min * CHORD / MU_AIR:,.0f} - {RHO_AIR * velocity_max * CHORD / MU_AIR:,.0f}'
        '</div>'
        '</div>'
    )

    st.markdown(
        reynolds_html,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# THIRD COLUMN: GEOMETRY ONLY
# ------------------------------------------------------------

with input_col_3:

    geometry_path = Path(__file__).resolve().parent / "GEOMETRY_DASHBOARD.PNG"

    if not geometry_path.is_file():
        st.error(
            f"Geometry image was not found: {geometry_path}. "
            "Keep GEOMETRY_DASHBOARD.PNG in the Dashboard folder."
        )
        st.stop()

    # Add vertical spacing to center the image in the column
    st.markdown(
        "<div style='height: 110px;'></div>",
        unsafe_allow_html=True,
    )

    st.image(
        str(geometry_path),
        use_container_width=True,
    )


# ============================================================
# CALCULATE BUTTON
# ============================================================

st.write("")

calculate = st.button(
    "Calculate Aerodynamic Performance",
    use_container_width=True,
)


# ============================================================
# CALCULATION
# ============================================================

if calculate:

    if aoa_min >= aoa_max:
        st.error("The minimum AoA must be smaller than the maximum AoA.")
        st.stop()

    number_of_intervals = (aoa_max - aoa_min) / aoa_step

    if not np.isclose(
        number_of_intervals,
        round(number_of_intervals),
        atol=1e-10,
    ):
        st.error(
            "The selected AoA interval must divide the requested AoA "
            "range exactly. For example, 0° to 20° with a 1° interval "
            "generates 21 prediction points from 0° to 20°."
        )
        st.stop()

    with st.spinner(
        "Starting MATLAB and calculating aerodynamic performance..."
    ):

        try:
            eng = get_matlab_engine()

            prediction = predict_polar(
                eng=eng,
                lambda_value=lambda_value,
                x=x_value,
                cq=cq_value,
                velocity=velocity,
                aoa_min=aoa_min,
                aoa_max=aoa_max,
                aoa_step=aoa_step,
            )

            st.session_state["prediction"] = prediction

        except Exception as error:
            st.error(
                f"An error occurred during the prediction:\n\n{error}"
            )
            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    # ========================================================
    # RESULTS SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Results Summary</div>',
        unsafe_allow_html=True,
    )

    # Convert predictions to NumPy arrays
    CL = np.asarray(prediction["CL"], dtype=float)
    CD = np.asarray(prediction["CD"], dtype=float)
    AOA = np.asarray(prediction["AoA"], dtype=float)

    # Find the maximum CL and its corresponding AoA
    max_cl_idx = np.argmax(CL)
    max_CL = CL[max_cl_idx]
    aoa_at_max_CL = AOA[max_cl_idx]

    # CD corresponding to the same AoA where CL is maximum
    CD_at_max_CL = CD[max_cl_idx]

    result_col_1, result_col_2 = st.columns(
        2,
        gap="medium",
    )

    with result_col_1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Maximum CL</div>
                <div class="result-value">{max_CL:.3f}</div>
                <div class="result-subtitle">
                    AoA = {aoa_at_max_CL:.1f}°
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col_2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">CD at Maximum CL</div>
                <div class="result-value">{CD_at_max_CL:.3f}</div>
                <div class="result-subtitle">
                    AoA = {aoa_at_max_CL:.1f}°
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # AERODYNAMIC POLARS
    # ========================================================

    st.markdown(
        '<div class="section-title">Aerodynamic Polars</div>',
        unsafe_allow_html=True,
    )

    plot_col_1, plot_col_2 = st.columns(2, gap="medium")

    with plot_col_1:
        fig_cl, ax_cl = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

        plot_polar(
            ax_cl,
            prediction["AoA"],
            prediction["CL"],
            r"Lift Coefficient, $C_L$",
            "Lift Coefficient Polar",
        )

        st.pyplot(fig_cl, use_container_width=True)
        plt.close(fig_cl)

    with plot_col_2:
        fig_cd, ax_cd = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

        plot_polar(
            ax_cd,
            prediction["AoA"],
            prediction["CD"],
            r"Drag Coefficient, $C_D$",
            "Drag Coefficient Polar",
        )

        st.pyplot(fig_cd, use_container_width=True)
        plt.close(fig_cd)



    # ========================================================
    # PREDICTED VALUES
    # ========================================================

    st.markdown(
        '<div class="section-title">Predicted Values</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
    prediction["results"].style
        .set_properties(**{
            "text-align": "center"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#00064D"),
                    ("color", "white"),
                    ("text-align", "center"),
                    ("font-weight", "bold"),
                ],
            }
        ]),
    use_container_width=True,
    hide_index=True,
)
