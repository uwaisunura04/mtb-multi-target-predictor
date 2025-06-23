# app.py

import streamlit as st
import deepchem as dc
import numpy as np
import json
import joblib
from rdkit import Chem
from deepchem.feat import ConvMolFeaturizer
from deepchem.models.torch_models import GCNModel

# Load featurizer
try:
    featurizer = joblib.load("featurizer.pkl")  # Use saved featurizer
except:
    featurizer = ConvMolFeaturizer()  # fallback if not found

# Load tasks
with open("task_list.json", "r") as f:
    tasks = json.load(f)

# Load GCN Model (must match saved one)
model = GCNModel(n_tasks=len(tasks), mode='classification', model_dir='graphconv_model')
model.restore()

# UI
st.title("🧪 MTB Drug Discovery: Virulence Protein Prediction")
st.markdown("Enter a SMILES string of a natural compound to predict its activity against MTB proteins.")

smiles = st.text_input("Enter SMILES:", value="CC1=C(C(=O)NC(=O)N1)N")

if st.button("Predict"):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # Featurize the SMILES
            features = featurizer.featurize([smiles])
            
            # Create dummy dataset for prediction
            dataset = dc.data.NumpyDataset(X=features)
            
            # Predict
            raw_preds = model.predict(dataset)
            prob = raw_preds[:, :, 1][0]  # Get class 1 probabilities
            pred = (prob > 0.5).astype(int)

            st.success("Prediction complete.")
            for t, p, pr in zip(tasks, pred, prob):
                st.write(f"**{t}**: {'🟢 Active' if p == 1 else '🔴 Inactive'} (prob: {pr:.2f})")
        except Exception as e:
            st.error(f"Something went wrong during prediction: {str(e)}")
    else:
        st.error("Invalid SMILES string.")