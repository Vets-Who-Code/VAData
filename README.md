# Veteran Suicide Prediction Model

## 📌 Project Overview
Veteran suicide is a critical issue, and this project leverages machine learning to analyze historical trends and predict future suicide rates among veterans. Using data from 2001-2022, the model considers year, geographic region, and state to estimate the number of veteran suicides.

## 📂 Dataset
The dataset is sourced from the U.S. Department of Veterans Affairs (VA) and contains:
- **Year:** 2001-2022
- **Geographic Region:** (Northeastern, Southern, etc.)
- **State:** All 50 U.S. states & territories
- **Number of Veteran Suicides**

## 🛠️ Tech Stack
- **Python** 🐍
- **TensorFlow / Keras** 🤖
- **Scikit-Learn** 📊
- **Pandas** 🏷️
- **Matplotlib** 📈

## 📑 Project Structure
```
/VAData
│── data/
│   ├── VA_State_Sheets_2001-2022_Appendix_508.xlsx   # Raw dataset
│── notebooks/
│   ├── veteran_suicide_analysis.ipynb   # Data analysis & model training
│── models/
│   ├── veteran_suicide_model.keras   # Saved trained model
│── src/
│   ├── train_model.py   # Train the model script
│   ├── predict.py   # Make predictions
│── README.md
│── requirements.txt
```

## 🚀 How to Run This Project
### 1️⃣ Set Up Environment
Clone the repository and install dependencies:
```sh
git clone https://github.com/yourusername/veteran-suicide-prediction.git
cd veteran-suicide-prediction
pip install -r requirements.txt
```

### 2️⃣ Train the Model
Run the training script to process the dataset and train the TensorFlow model:
```sh
python src/train_model.py
```

### 3️⃣ Make Predictions
After training, make predictions for a specific year and state:
```sh
python src/predict.py --year 2025 --state "Texas"
```
#### Example Output:
```
Predicted Veteran Suicides in Texas (2025): 105
```

## 📊 Model Performance
- **Mean Absolute Error (MAE):** 232.97 (Average error in suicide predictions)
- **Mean Squared Error (MSE):** 349,559.59 (Overall prediction accuracy measure)
- **Training Improvement:** Loss decreased over 100 epochs, indicating learning.

## 🔍 Future Improvements
- **Feature Expansion:** Incorporate factors like GDP, unemployment rates, and VA funding per state.
- **Model Optimization:** Experiment with different architectures, layers, and optimizers.
- **Deployment:** Implement a FastAPI endpoint for real-time predictions.
- **Dashboard Integration:** Use Streamlit to visualize veteran suicide trends.

## 📜 License
This project is open-source under the **MIT License**.

## 👥 Contributing
Contributions are welcome! Open an issue or submit a pull request if you find bugs or have ideas to improve the model.

---

### 💡 Acknowledgment
This project is dedicated to supporting veterans and raising awareness about mental health challenges within the veteran community.

