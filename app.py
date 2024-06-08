import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import joblib

# Function to load models and preprocessing details
def load_models_and_details(filepath):
    try:
        model_data = joblib.load(filepath)
        return model_data['models'], model_data['model_names'], model_data['preprocessing_details']
    except FileNotFoundError:
        st.error("Could not find model data file.")
        return None, None, None
    except KeyError as e:
        st.error(f"Missing key in model data file: {e}")
        return None, None, None

# Function to preprocess input data
def preprocess_input(input_data, X_scaler):
    input_smooth = savgol_filter(input_data, window_length=11, polyorder=2, axis=1)
    input_scaled = X_scaler.transform(input_smooth)
    return input_scaled

# Function to predict composition
def predict_composition(input_data, preprocessing_details, best_models):
    X_scaler = preprocessing_details['X_scaler']
    y_scaler = preprocessing_details['y_scaler']
    indices_list = preprocessing_details['indices']
    
    predictions = []
    for i, model in enumerate(best_models):
        indices = indices_list[i]
        input_scaled = preprocess_input(input_data, X_scaler)
        input_selected = input_scaled[:, indices]

        y_pred = model.predict(input_selected)
        
        # Handle reshaping based on the dimension of the prediction
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)  # Reshape to 2D if it's a single-dimensional array
        
        predictions.append(y_pred.flatten())
    
    predictions_array = np.column_stack(predictions)
    predictions_original_scale = y_scaler.inverse_transform(predictions_array)
    return predictions_original_scale

# Main execution
st.title("Spectral Data Prediction Interface")

models, model_names, preprocessing_details = load_models_and_details('model_data.pkl')

if models is not None and preprocessing_details is not None:
    uploaded_file = st.file_uploader("Upload your spectral data CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file).values
        if input_data.shape[1] != 18:  # Assuming 18 is the number of spectral features expected
            st.error(f"Uploaded data has {input_data.shape[1]} features, but 18 are expected.")
        else:
            predicted_composition = predict_composition(input_data, preprocessing_details, models)
            st.write("Predicted Composition:")
            labels = ['PH', 'EC', 'OC', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Cu', 'Zn', 'B']
            predicted_df = pd.DataFrame(predicted_composition, columns=labels)
            st.write(predicted_df)
            st.write("Models used for prediction:")
            st.write(model_names)
    

