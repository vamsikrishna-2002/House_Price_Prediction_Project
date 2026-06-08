"""
Flask web application for house price prediction
"""
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# State price adjustment factors (relative to California baseline)
# Based on average home prices by state (2023-2024 data)
STATE_ADJUSTMENT_FACTORS = {
    'AL': 0.35, 'AK': 0.65, 'AZ': 0.55, 'AR': 0.32, 'CA': 1.00,
    'CO': 0.75, 'CT': 0.85, 'DE': 0.60, 'FL': 0.50, 'GA': 0.40,
    'HI': 1.20, 'ID': 0.50, 'IL': 0.45, 'IN': 0.35, 'IA': 0.35,
    'KS': 0.35, 'KY': 0.35, 'LA': 0.35, 'ME': 0.50, 'MD': 0.70,
    'MA': 0.90, 'MI': 0.40, 'MN': 0.45, 'MS': 0.30, 'MO': 0.35,
    'MT': 0.50, 'NE': 0.35, 'NV': 0.55, 'NH': 0.65, 'NJ': 0.80,
    'NM': 0.40, 'NY': 0.85, 'NC': 0.40, 'ND': 0.40, 'OH': 0.35,
    'OK': 0.32, 'OR': 0.65, 'PA': 0.40, 'RI': 0.60, 'SC': 0.38,
    'SD': 0.40, 'TN': 0.38, 'TX': 0.40, 'UT': 0.60, 'VT': 0.55,
    'VA': 0.55, 'WA': 0.75, 'WV': 0.30, 'WI': 0.40, 'WY': 0.45
}

# Load the trained model and feature names
model_path = 'house_price_model.pkl'
feature_names_path = 'feature_names.pkl'

# Check if model exists, if not, train it
if not os.path.exists(model_path):
    print("Model not found. Training model...")
    from train_model import train_and_save_model
    train_and_save_model()

# Load model and feature names
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(feature_names_path, 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form
        
        # Get state selection
        state = data.get('State', 'CA')
        state_adjustment = STATE_ADJUSTMENT_FACTORS.get(state, 1.0)
        
        # Extract features in the correct order (excluding Latitude and Longitude)
        # Note: CityName and StreetName are collected but not used in prediction
        # Using default Population value (median ~1166) since city/street are text fields
        features = [
            float(data.get('MedInc', 0)),
            float(data.get('HouseAge', 0)),
            float(data.get('AveRooms', 0)),
            float(data.get('AveBedrms', 0)),
            float(data.get('Population', 1166)),  # Default to median population
            float(data.get('AveOccup', 0))
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction (based on California model)
        prediction = model.predict(features_array)[0]
        
        # Convert numpy types to Python native types for JSON serialization
        prediction = float(prediction)
        
        # Apply house age adjustment (newer houses should be more expensive than older houses)
        house_age = float(data.get('HouseAge', 0))
        # Calculate age adjustment: newer houses get premium, older houses get discount
        # Formula: premium for new (age 0-10), then decreasing, discount for old (age 30+)
        # Maximum premium: +30% for brand new houses (age 0)
        # Maximum discount: -40% for very old houses (age 50+)
        if house_age <= 10:
            # New houses (0-10 years): premium decreases linearly from 30% to 0%
            age_adjustment = 1.0 + (10 - house_age) * 0.03  # 1.30 to 1.00
        elif house_age <= 30:
            # Middle-aged houses (11-30 years): slight discount
            age_adjustment = 1.0 - (house_age - 10) * 0.01  # 1.00 to 0.80
        else:
            # Old houses (31+ years): larger discount, capped at 40%
            age_adjustment = max(0.60, 0.80 - (house_age - 30) * 0.01)  # 0.80 to 0.60
        
        prediction = prediction * age_adjustment
        
        # Apply state adjustment factor
        prediction = prediction * state_adjustment
        
        # The prediction is in hundreds of thousands of dollars
        # Convert to actual dollar amount
        price_in_dollars = prediction * 100000
        
        # Calculate additional data for charts
        house_age = float(data.get('HouseAge', 0))
        base_prediction = model.predict(features_array)[0]
        base_price = float(base_prediction) * 100000
        
        # Calculate age impact data for chart
        age_impact_data = []
        for age in range(0, 51, 5):
            if age <= 10:
                age_adj = 1.0 + (10 - age) * 0.03
            elif age <= 30:
                age_adj = 1.0 - (age - 10) * 0.01
            else:
                age_adj = max(0.60, 0.80 - (age - 30) * 0.01)
            adjusted_price = base_price * age_adj * state_adjustment
            age_impact_data.append({'age': age, 'price': round(adjusted_price, 2)})
        
        # Calculate state comparison data (top 10 states)
        top_states = ['CA', 'HI', 'MA', 'NY', 'NJ', 'CT', 'WA', 'CO', 'MD', 'NH']
        state_comparison_data = []
        for st in top_states:
            st_adj = STATE_ADJUSTMENT_FACTORS.get(st, 1.0)
            st_price = base_price * age_adjustment * st_adj
            state_comparison_data.append({
                'state': st,
                'price': round(st_price, 2)
            })
        
        # Format the response
        result = {
            'success': True,
            'prediction': round(prediction, 4),
            'price_in_dollars': round(price_in_dollars, 2),
            'formatted_price': f"${price_in_dollars:,.2f}",
            'state': state,
            'base_price': round(base_price, 2),
            'age_adjustment': round(age_adjustment, 4),
            'state_adjustment': round(state_adjustment, 4),
            'age_impact_data': age_impact_data,
            'state_comparison_data': state_comparison_data,
            'house_age': house_age,
            'features': {
                'MedInc': float(data.get('MedInc', 0)),
                'HouseAge': house_age,
                'AveRooms': float(data.get('AveRooms', 0)),
                'AveBedrms': float(data.get('AveBedrms', 0)),
                'AveOccup': float(data.get('AveOccup', 0))
            }
        }
        
        return jsonify(result) if request.is_json else render_template(
            'index.html', 
            prediction=result['formatted_price'],
            prediction_raw=result['prediction']
        )
        
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        return jsonify({'success': False, 'error': error_msg}) if request.is_json else render_template(
            'index.html',
            error=error_msg
        )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

