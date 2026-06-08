# House Price Prediction Website

A web application for predicting house prices using machine learning. This application uses the California Housing dataset and XGBoost regression model to predict median house values.

## Features

- 🏠 Interactive web interface for house price prediction
- 🤖 Machine learning model powered by XGBoost
- 📊 Predictions based on 8 key features:
  - Median Income
  - House Age
  - Average Rooms
  - Average Bedrooms
  - Population
  - Average Occupancy
  - Latitude
  - Longitude

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model (if not already trained):**
   ```bash
   python train_model.py
   ```
   This will create `house_price_model.pkl` and `feature_names.pkl` files.

2. **Run the Flask application:**
   ```bash
   python app.py
   ```

3. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

4. **Fill in the form with house features and click "Predict House Price"**

## Model Information

- **Algorithm:** XGBoost Regressor
- **Dataset:** California Housing Dataset (20,640 samples)
- **Features:** 8 numerical features
- **Target:** Median house value (in hundreds of thousands of dollars)

## Example Input Values

- Median Income: 8.3
- House Age: 41
- Average Rooms: 6.98
- Average Bedrooms: 1.02
- Population: 322
- Average Occupancy: 2.56
- Latitude: 37.88
- Longitude: -122.23

## Project Structure

```
House prediction/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── house_price_model.pkl  # Trained model (generated)
├── feature_names.pkl      # Feature names (generated)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
└── README.md             # This file
```

## Notes

- The model predicts prices in hundreds of thousands of dollars (e.g., 2.5 = $250,000)
- The model is trained on California housing data, so predictions are most accurate for California properties
- Latitude should be between 32 and 42 (California range)
- Longitude should be between -125 and -114 (California range)

## License

This project is open source and available for educational purposes.

