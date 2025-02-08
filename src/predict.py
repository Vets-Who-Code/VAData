import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load model and encoders
model = tf.keras.models.load_model("models/veteran_suicide_model.keras")

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
state_encoder = encoders["state"]
region_encoder = encoders["region"]

# Function to predict veteran suicides
def predict_suicides(year, state):
    try:
        state_encoded = state_encoder.transform([state])[0]
    except ValueError:
        print(f"Error: State '{state}' not found in training data.")
        return

    # Placeholder for region (since it's unknown in user input)
    region_encoded = 0  # Default value (modify as needed)

    # Prepare input
    X_input = np.array([[year, region_encoded, state_encoded]])

    # Make prediction
    predicted_suicides = model.predict(X_input)[0][0]
    print(f"Predicted Veteran Suicides in {state} ({year}): {int(predicted_suicides)}")

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict veteran suicides based on year and state")
    parser.add_argument("--year", type=int, required=True, help="Year for prediction")
    parser.add_argument("--state", type=str, required=True, help="State name for prediction")

    args = parser.parse_args()
    predict_suicides(args.year, args.state)