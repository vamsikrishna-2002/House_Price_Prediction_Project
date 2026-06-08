"""
Script to train and save the house price prediction model
"""
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import os

def train_and_save_model():
    """Train the XGBoost model and save it"""
    print("Loading California Housing dataset...")
    house_price_dataset = sklearn.datasets.fetch_california_housing()
    
    # Loading the dataset to a pandas dataframe
    house_price_dataframe = pd.DataFrame(
        house_price_dataset.data, 
        columns=house_price_dataset.feature_names
    )
    
    # Add the target column to the dataframe
    house_price_dataframe['price'] = house_price_dataset.target
    
    # Splitting the data and target
    # Exclude Latitude and Longitude features
    X = house_price_dataframe.drop(['price', 'Latitude', 'Longitude'], axis=1)
    Y = house_price_dataframe['price']
    
    # Splitting the data into training data and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    
    print("Training XGBoost model...")
    # Load and train the model
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    
    # Evaluate the model
    training_data_prediction = model.predict(X_train)
    test_data_prediction = model.predict(X_test)
    
    from sklearn import metrics
    score_1 = metrics.r2_score(Y_train, training_data_prediction)
    score_2 = metrics.r2_score(Y_test, test_data_prediction)
    
    print(f"R2 Score on Training data: {score_1}")
    print(f"R2 Score on Test data: {score_2}")
    
    # Save the model
    model_filename = 'house_price_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved as {model_filename}")
    
    # Save feature names for later use
    feature_names = list(X.columns)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model training completed successfully!")
    return model

if __name__ == "__main__":
    train_and_save_model()

