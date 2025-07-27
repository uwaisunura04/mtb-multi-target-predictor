import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load trained model
model = joblib.load("regression_model.pkl")  # Make sure this file is in the same directory

# Define MTB protein targets
mtb_targets = ["InhA", "KatG", "GyrA", "EmbB", "RpoB", "PncA", "DprE1", "EthA", "KasA", "Rv1636"]  # Extend as needed

# Molecular descriptor function
def featurize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ]
    except:
        return [np.nan, np.nan, np.nan, np.nan]

# Streamlit UI
st.set_page_config(page_title="MTB Drug Activity Predictor", layout="wide")
st.title("🧪 TB Drug Activity Prediction App 🇳🇬")
st.write("This app predicts the **bioactivity (pIC50)** of compounds against MTB target proteins using a trained ML regression model.")

# Input section
st.subheader("🔬 Input SMILES")
smiles_input = st.text_area("Enter one or more SMILES (one per line)", height=200)

if st.button("🧠 Predict Activity"):
    if smiles_input.strip() == "":
        st.warning("⚠️ Please enter at least one SMILES string.")
    else:
        smiles_list = smiles_input.strip().split("\n")
        results = []

        for smiles in smiles_list:
            features = featurize(smiles)
            if np.nan in features:
                st.error(f"❌ Invalid SMILES: {smiles}")
                continue

            for target in mtb_targets:
                # You can optionally add target-specific features here
                prediction = model.predict([features])[0]
                results.append({
                    "Target Protein": target,
                    "SMILES": smiles,
                    "Predicted pIC50": round(prediction, 3)
                })

        if results:
            df_results = pd.DataFrame(results)
            st.success("✅ Prediction Complete")
            st.dataframe(df_results, use_container_width=True)

            # Download option
            csv = df_results.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "mtb_predictions.csv", "text/csv")
