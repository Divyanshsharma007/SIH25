from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Try to import catboost with fallback
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available, using fallback mode")

# Load model
model = None

def load_model():
    global model
    try:
        model_path = os.path.join('..', 'models', 'catboost-model.pkl')
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        print("✅ CatBoost model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

# Load model when service starts
load_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK' if model is not None else 'ERROR',
        'model_loaded': model is not None,
        'catboost_available': CATBOOST_AVAILABLE
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
            
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'Features array is required'}), 400
            
        features = data['features']
        
        if len(features) != 35:  # Adjust based on your actual feature count
            return jsonify({
                'error': f'Expected 35 features, got {len(features)}'
            }), 400

        # Define your dataset's feature names (must match training)
        # Replace with your actual feature names
        feature_names = [
            'Marital status', 'Application mode', 'Application order', 'Course',
            'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
            "Mother's qualification", "Father's qualification", "Mother's occupation",
            "Father's occupation", 'Displaced', 'Educational special needs', 
            'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
            'Age at enrollment', 'International', 'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
            'Inflation rate', 'GDP', 'Application year'
        ]

        df = pd.DataFrame([features], columns=feature_names)

        prediction = model.predict(df)
        probabilities = model.predict_proba(df)

        class_names = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}

        return jsonify({
            'prediction': class_names[prediction[0]],
            'probabilities': {
                class_names[i]: float(probabilities[0][i])
                for i in range(len(class_names))
            }
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Python Prediction Service...")
    print(f"Python version: {sys.version}")
    app.run(host='0.0.0.0', port=5000, debug=True)