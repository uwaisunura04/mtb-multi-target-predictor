import streamlit as st
import deepchem as dc
import numpy as np
import os

st.title("Molecule Bioactivity Prediction (GraphConvModel)")

# Model directory
model_dir = "graphcov_model"  # ✅ make sure this is the correct name

# Debug: list files in the working directory
st.write("Files in current directory:", os.listdir())

# Try to load model
try:
    if not os.path.exists(model_dir):
        st.error(f"Model directory '{model_dir}' not found!")
        model = None
    else:
        model = dc.models.GraphConvModel(n_tasks=1, mode='regression', model_dir=model_dir)
        model.restore()
        st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    model = None

# SMILES input
smiles_input = st.text_input("Enter a SMILES string:")

# Prediction
if smiles_input and model:
    try:
        featurizer = dc.feat.ConvMolFeaturizer()
        mol = featurizer.featurize([smiles_input])
        prediction = model.predict(mol)
        st.success(f"🔮 Predicted activity: {prediction[0][0]:.4f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
elif smiles_input and not model:
    st.warning("⚠️ Model not available. Cannot predict.")
