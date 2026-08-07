import pandas as pd
import joblib
import os
import joblib

def load_prediction_artifacts():
    # Dynamically find the absolute path of the directory this script lives in (dashboard/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root directory
    project_root = os.path.dirname(current_script_dir)
    
    # Create foolproof absolute paths to your models directory
    model_path = os.path.join(project_root, 'models', 'churn_model.pkl')
    preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
    
    # Load your artifacts using the absolute paths
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor
    
def predict_single_customer(input_data):
    model, preprocessor = load_prediction_artifacts()
    input_df = pd.DataFrame([input_data])
    processed_input = preprocessor.transform(input_df)
    prob = model.predict_proba(processed_input)[0][1]
    
    if prob >= 0.70:
        risk_tier = "High Risk"
    elif prob >= 0.40:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "Low Risk"
        
    return risk_tier, prob