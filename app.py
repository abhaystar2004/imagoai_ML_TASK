import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Import pandas for better CSV handling

# Load the saved model, scaler, and PCA
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Streamlit app title
st.title("Random Forest Prediction App")

# Allow the user to upload a CSV file with 448 features
uploaded_file = st.file_uploader("Upload a CSV file with 448 features", type=["csv"])

if uploaded_file is not None:
  try:
    # Read the uploaded file using pandas
    df = pd.read_csv(uploaded_file)

    # Drop non-feature columns (e.g., 'hsi_id' and 'vomitoxin_ppb')
    if 'hsi_id' in df.columns:
        df = df.drop(columns=['hsi_id'])
    if 'vomitoxin_ppb' in df.columns:
        df = df.drop(columns=['vomitoxin_ppb'])

    # Convert the DataFrame to a NumPy array
    input_data = df.values

    # Ensure the input has the correct shape
    if input_data.shape[1] != 448:
        st.error("The uploaded file must have exactly 448 features (excluding non-feature columns).")
    else:
        # Scale and transform the input
        scaled_features = scaler.transform(input_data)
        pca_features = pca.transform(scaled_features)
        predictions = rf_model.predict(pca_features)

        # Display predictions for all rows
        df['Predicted Class'] = predictions
        st.success("Predictions completed successfully!")
        st.write(df["Predicted Class"])  # Display the DataFrame with predictions
  except Exception as e:
      st.error(f"An error occurred while processing the file: {e}")