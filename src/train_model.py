import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "data/VA_State_Sheets_2001-2022_Appendix_508.xlsx"
df_state = pd.read_excel(file_path, sheet_name="Veteran Suicides by State", skiprows=1)

# Rename columns
df_state.columns = ["Year", "Geographic Region", "State", "Veteran Suicides"]

# Encode categorical features
state_encoder = LabelEncoder()
df_state["State"] = state_encoder.fit_transform(df_state["State"])

region_encoder = LabelEncoder()
df_state["Geographic Region"] = region_encoder.fit_transform(df_state["Geographic Region"])

# Define features and target
X = df_state[["Year", "Geographic Region", "State"]]
y = df_state["Veteran Suicides"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert target to numeric and handle missing values
y_train = pd.to_numeric(y_train, errors="coerce").fillna(y_train.median()).astype(np.float32)
y_test = pd.to_numeric(y_test, errors="coerce").fillna(y_test.median()).astype(np.float32)

# Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save("models/veteran_suicide_model.keras")

# Save encoders for prediction
import pickle
with open("models/encoders.pkl", "wb") as f:
    pickle.dump({"state": state_encoder, "region": region_encoder}, f)

print("Model training complete. Saved to models/veteran_suicide_model.keras")