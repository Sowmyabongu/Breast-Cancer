import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data_path = "data.csv"  # Ensure this is the correct path to your uploaded file
df = pd.read_csv("C:\Users\sowmy\Downloads\ML project\data.csv")

# Preprocess dataset (assuming target column is 'diagnosis')
X = df.drop(columns=['diagnosis'])  # Features
y = df['diagnosis'].map({'B': 0, 'M': 1})  # Convert labels to 0 (benign) and 1 (malignant)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "tumor_model.sav")

# Load trained model
tumor_model = joblib.load("tumor_model.sav")

# Streamlit UI
st.title("Tumor Classification App")
st.write("Enter tumor characteristics to predict if it's benign or malignant.")

# Create input fields for all features
user_input = []
columns = X.columns

for col in columns:
    value = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))
    user_input.append(value)

# Predict button
if st.button("Predict Tumor Type"):
    prediction = tumor_model.predict([user_input])
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.success(f"The predicted tumor type is: {result}")
