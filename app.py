import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load the model
model = joblib.load("MTB_RF_model.pkl")

# Molecular descriptor calculation
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
    ]

# App UI
st.title("MTB Drug Target IC50 Predictor")
st.write("Built for Nigeria 🇳🇬 using RDKit + Machine Learning")

# SMILES input
smiles_input = st.text_area("Enter SMILES (one per line)", height=150)

if st.button("Predict IC50"):
    smiles_list = smiles_input.strip().split("\n")
    results = []
    
    for smi in smiles_list:
        desc = compute_descriptors(smi)
        if desc:
            prediction = model.predict([desc])[0]
            results.append({"SMILES": smi, "Predicted IC50 (nM)": round(prediction, 2)})
        else:
            results.append({"SMILES": smi, "Predicted IC50 (nM)": "Invalid SMILES"})

    df = pd.DataFrame(results)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv", "text/csv")
