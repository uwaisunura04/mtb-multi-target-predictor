import streamlit as st
import deepchem as dc
import numpy as np

# Set title
st.title("Molecule Bioactivity Prediction (GraphConvModel)")

# Load the model
model_dir = "graphcov_model"  # Make sure this is the correct relative path to your saved model
model = dc.models.GraphConvModel(n_tasks=1, mode='regression', model_dir=model_dir)
model.restore()

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