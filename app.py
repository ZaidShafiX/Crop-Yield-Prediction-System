"""
Crop Yield Prediction System - Streamlit web app
------------------------------------------------
Loads the model saved by the notebook (Step 10.3) and predicts the crop yield
from the 5 field conditions: Fertilizer, temp, N, P, K.
"""

import os
import pickle

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MODEL_FILE = "crop_yield_model_full_dataset.pkl"

# Mean 10-fold cross-validation RMSE of the best algorithm.
# Copy this number from Step 10.1 of the notebook ("Expected RMSE").
CV_RMSE = 0.171

# Feature name -> (label, min, max, default, step).
# The defaults are a realistic field (close to many rows of the training data).
# Min/max are the ranges seen in the training data (rounded inward to 1 decimal),
# so the app never asks the model to predict outside what it has learned from.
FEATURES = {
    "Fertilizer": ("Fertilizer", 49.8, 80.2, 73.0, 0.1),
    "temp": ("Temperature", 23.8, 40.2, 29.0, 0.1),
    "N": ("Nitrogen (N)", 58.9, 80.2, 74.0, 0.1),
    "P": ("Phosphorus (P)", 17.8, 25.1, 21.0, 0.1),
    "K": ("Potassium (K)", 14.8, 22.0, 19.0, 0.1),
}

# Real rows of the training data, so users can start from a realistic field.
# label -> ((Fertilizer, temp, N, P, K), yield recorded in the dataset)
EXAMPLES = {
    "High yield field": ((80.0, 28.0, 80.0, 24.0, 20.0), "12"),
    "Good yield field": ((71.0, 28.0, 77.0, 21.0, 20.0), "11"),
    "Medium yield field": ((72.0, 29.0, 73.0, 21.0, 19.0), "10"),
    "Hot, lower yield field": ((58.0, 37.0, 66.0, 22.0, 18.0), "8"),
    "Low yield field": ((60.0, 39.0, 60.0, 18.0, 15.0), "5.5 - 6"),
}
NO_EXAMPLE = "(choose an example)"

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")


# ----------------------------------------------------------------------------
# Load the trained model once (cached between interactions)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_FILE}' was not found. Put it in the same folder as app.py "
        "(and in the GitHub repository when deploying)."
    )
    st.stop()
except Exception as error:  # most often a scikit-learn version mismatch
    st.error(
        "The model file could not be loaded. This usually means the scikit-learn version "
        "used here is different from the one that trained the model. Set the same version "
        "in requirements.txt (the notebook prints it in Step 1).\n\n"
        f"Details: {type(error).__name__}: {error}"
    )
    st.stop()

algorithm_name = model.named_steps["regressor"].__class__.__name__

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("🌾 Crop Yield Predictor")
st.write(
    "Enter the field conditions and the model will predict the **crop yield**. "
    "Machine Learning Fundamentals (AIC354) - Assignment 3."
)

# Slider values live in st.session_state (key = feature name) so an example can fill them in.
for name, (_, _, _, default, _) in FEATURES.items():
    st.session_state.setdefault(name, default)


def apply_example():
    choice = st.session_state["example_choice"]
    if choice != NO_EXAMPLE:
        for name, value in zip(FEATURES, EXAMPLES[choice][0]):
            st.session_state[name] = value


st.selectbox(
    "Start from a real field from the training data (optional)",
    [NO_EXAMPLE, *EXAMPLES],
    key="example_choice",
    on_change=apply_example,
)

with st.form("field_form"):
    st.subheader("Field conditions")
    left, right = st.columns(2)
    values = {}
    for position, (name, (label, low, high, default, step)) in enumerate(FEATURES.items()):
        column = left if position % 2 == 0 else right
        values[name] = column.slider(label, min_value=low, max_value=high, step=step, key=name)
    submitted = st.form_submit_button("Predict yield", type="primary")

if submitted:
    # Same columns, same order as in the notebook (Step 11.2)
    feature_vector = pd.DataFrame([values])[list(FEATURES)]
    predicted_yield = float(model.predict(feature_vector)[0])

    st.subheader("Prediction")
    st.metric("Predicted crop yield", f"{predicted_yield:.2f}")
    st.caption(
        f"Typical error: about ± {CV_RMSE:.2f} for realistic field conditions "
        f"(10-fold cross-validation RMSE of the {algorithm_name} model). "
        "For unusual combinations of inputs the error can be larger."
    )
    for example_label, (example_values, recorded) in EXAMPLES.items():
        if all(abs(values[n] - v) < 0.05 for n, v in zip(FEATURES, example_values)):
            st.info(
                f"These are the conditions of a real row of the training data (recorded yield: {recorded}). "
                "A close match is expected here; it shows that the app works, not how well the model "
                "predicts for new fields."
            )
            break

with st.expander("About this model"):
    st.markdown(
        f"""
- **Algorithm:** {algorithm_name} (best of 5 algorithms compared with 10-fold cross validation)
- **Inputs:** Fertilizer, temperature, soil Nitrogen (N), Phosphorus (P) and Potassium (K)
- **Output:** crop yield (same unit as the `yeild` column of the dataset; the unit is not documented in the file)
- **Training data:** crop yield dataset with soil and weather values (2,585 unique rows)
- **Input limits:** the sliders are limited to the value ranges seen in the training data,
  because the model cannot predict reliably outside them.
- **Input combinations:** the sliders move independently, but in real fields the five values
  are strongly related to each other. A combination that is unusual (for example, very high
  fertilizer together with very high temperature) can be far from every training example even
  when each value is inside its slider range. For such combinations this model tends to answer with
  a value close to the average yield (about 8.5), and the typical error shown above does not apply.
- **Caution:** the dataset contains only a small number of distinct field conditions,
  so the model is most accurate for conditions similar to those in the data.
"""
    )
