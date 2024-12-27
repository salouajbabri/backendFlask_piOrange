from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load phone numbers from the uploaded file
data_file = "data.xlsx"
data = pd.read_excel(data_file)

data = pd.read_excel(data_file, dtype={'Numéro': str})  # Ensure 'Numéro' is read as string

# Convert phone numbers to a list for quick lookup
phone_numbers = data['Numéro'].astype(str).tolist() 

# Function to load models
def load_models():
    model_path = "model"
    print("Looking for models in:", os.path.abspath(model_path))  # Debugging line

    ent_model = joblib.load(os.path.join(model_path, 'ent_model.pkl'))
    offer_model = joblib.load(os.path.join(model_path, 'offer_model.pkl'))
    mmr_model = joblib.load(os.path.join(model_path, 'mmr_model.pkl'))
    pref_model = joblib.load(os.path.join(model_path, 'pref_model.pkl'))
    feature_columns = joblib.load(os.path.join(model_path, 'model_features.pkl'))
    return ent_model, offer_model, mmr_model, pref_model, feature_columns

# Function to map MMR values to offers
def map_to_offer(mmr_value):
    if mmr_value <= 49:
        return "Forfait Yo 49 Dh - 1 Go / 5 heures"
    elif mmr_value <= 99:
        return "Forfait Yo 99 Dh - 11 Go / 11 heures"
    elif mmr_value <= 199:
        return "Forfait Yo 199 Dh - 40 Go / 5 heures"
    elif mmr_value <= 299:
        return "Forfait Yoxo 99 Dh - 20 Go / 2 heures / SMS illimités"
    else:
        return "Custom High-Tier Plan"

# Function to map predicted services to service names
def map_to_service(predicted):
    service_mapping = {
        0: "Spotify, Playvod",
        1: "Gameloft, Freefire",
        2: "Shahid, LaLiga Xtra",
        3: "Fitness24, Manhaj"
    }
    services = []
    for idx, val in enumerate(predicted):
        if val == 1 and idx in service_mapping:
            services.append(service_mapping[idx])
    return ", ".join(services) if services else "No Preferences"

# Load pre-trained models
ent_model, offer_model, mmr_model, pref_model, feature_columns = load_models()

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    try:
        request_data = request.json
        numero = str(request_data.get("numero"))  # Get the phone number from the request

        if numero in phone_numbers:
            # Retrieve the user's name from the Excel file
            user_row = data[data['Numéro'] == numero]
            user_name = user_row['Nom'].values[0]  # Assuming the 'Nom' column exists

            return jsonify({"success": True, "message": "Login successful!", "name": user_name}), 200
        else:
            return jsonify({"success": False, "message": "Invalid phone number"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        features = [
            input_data['Voix_minutes'], 
            input_data['Data_MO'], 
            input_data['Youtube'],
            input_data['Navigation'], 
            input_data['Gaming'], 
            input_data['Musique'],
            input_data['Reseaux_sociaux']
        ]

        # Reshape input and predict
        input_df = pd.DataFrame([features], columns=feature_columns)
        mmr_pred = mmr_model.predict(input_df)[0]
        service_pred = pref_model.predict(input_df)[0]

        # Map results
        offer = map_to_offer(mmr_pred)
        services = map_to_service(service_pred)

        # Return response
        return jsonify({
            "MMR": round(mmr_pred, 2),
            "Best Offer": offer,
            "Desired Services": services
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True)
