import streamlit as st
import deepchem as dc
import numpy as np
import os

st.title("Molecule Bioactivity Prediction (GraphConvModel)")

# Display working directory contents
st.write("Files in current directory:", os.listdir())

# Try to load the model
try:
    model_dir = "model"
    if not os.path.exists(model_dir):
        st.error("Model directory not found!")
    else:
        model = dc.models.GraphConvModel(n_tasks=1, mode='regression', model_dir=model_dir)
        model.restore()
        st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Input form
smiles_input = st.text_input("Enter a SMILES string:")

# Predict
if smiles_input:
    try:
        featurizer = dc.feat.ConvMolFeaturizer()
        mol = featurizer.featurize([smiles_input])
        prediction = model.predict(mol)
        st.success(f"Predicted activity: {prediction[0][0]:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
