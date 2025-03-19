import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load datasets
exercise = pd.read_csv("exercise.csv")
calories = pd.read_csv("calories.csv")

# Merge datasets
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Calculate BMI
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Encode Gender column
exercise_df["Gender"] = exercise_df["Gender"].apply(lambda x: 1 if x == "Male" else 0)

# Prepare features and labels
features = exercise_df[["Age", "Height", "Weight", "Duration", "Heart_Rate", "Gender"]]
labels = exercise_df[["Calories"]]  # Assuming Calories is the target for prediction

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

# Train the Decision Tree model
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)

# Make predictions on test data
y_pred = decision_tree.predict(X_test)

# Calculate accuracy metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Function to calculate step count based on duration
def estimate_step_count(duration):
    # Assume an average of 100 steps per minute
    steps_per_minute = 100
    estimated_steps = duration * steps_per_minute
    return estimated_steps

# Apply custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        color: #0d3abf;
        text-align: center;
        font-weight: bold;
        background-color: #F0E68C;
        padding: 10px;
        border-radius: 10px;
    }
    .prediction-header {
        color: #4682B4;
        font-size: 30px;
        font-weight: bold;
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 18px;
        color: #9ede31;
        font-weight: bold;
    }
    .stNumberInput input {
        border: 2px solid #4682B4;
    }
    .stButton button {
        background-color: #1cedea;
        color: #1cedea;
        font-size: 18px;
        border-radius: 8px;
    }
    body {
        background-color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<div class="main-title"> Personal Fitness Tracker App </div>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:24px; font-weight:bold; color:#4682B4;">Track. Improve. Succeed!</p>',
    unsafe_allow_html=True
)

# User input for prediction
def user_input_features():
    age = st.number_input("Age: ", min_value=10, max_value=100, value=30)
    height = st.number_input("Height (cm): ", min_value=140, max_value=220, value=170)
    weight = st.number_input("Weight (kg): ", min_value=30, max_value=150, value=70)
    duration = st.number_input("Duration (min): ", min_value=0, max_value=120, value=30)
    heart_rate = st.number_input("Heart Rate: ", min_value=60, max_value=180, value=80)
    gender = st.radio("Gender: ", ("Male", "Female"))
    
    gender_encoded = 1 if gender == "Male" else 0

    data = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Gender": gender_encoded
    }
    
    return pd.DataFrame(data, index=[0])

user_data = user_input_features()

# Make predictions
predicted_calories = decision_tree.predict(user_data)

# Calculate BMI
predicted_bmi = user_data["Weight"].values[0] / ((user_data["Height"].values[0] / 100) ** 2)

# Estimate step count
predicted_steps = estimate_step_count(user_data["Duration"].values[0])

# Determine exercise intensity
def determine_exercise_intensity(heart_rate):
    if heart_rate < 80:
        return "Low Intensity"
    elif 80 <= heart_rate < 140:
        return "Moderate Intensity"
    else:
        return "High Intensity"

predicted_intensity = determine_exercise_intensity(user_data["Heart_Rate"].values[0])

# Display predictions with a colorful header
st.write("---")
st.markdown('<div class="prediction-header">Predicted Values:</div>', unsafe_allow_html=True)
st.write(f"**Calories Burned:** {round(predicted_calories[0], 2)} kilocalories")
st.write(f"**BMI:** {round(predicted_bmi, 2)}")
st.write(f"**Estimated Step Count:** {predicted_steps} steps")
st.write(f"**Exercise Intensity:** {predicted_intensity}")

# Display Model Accuracy
print(f"Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Add a footer with additional details
st.write("🔍 *Note: The estimates are based on general data and may vary based on individual physiology.*")
