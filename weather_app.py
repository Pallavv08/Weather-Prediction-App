import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


st.set_page_config(page_title="🌧️ Weather Prediction", layout="centered")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('weather.csv')
        df = df.sample(5000, random_state=42)  
        df = df.dropna()  
        return df
    except FileNotFoundError:
        st.error("Error: 'weather.csv' file not found. Please download it from Kaggle.")
        st.stop()

df = load_data()

# model training

@st.cache_resource
def train_model():
    X = df[["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm", "WindSpeed3pm", "Pressure3pm"]]
    y = df["RainTomorrow"].map({"Yes": 1, "No": 0}) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# UI

st.title("Weather Prediction")
st.write("Predict whether it will rain tomorrow based on weather data.")


st.header("Weather Input Features")
min_temp = st.slider("Min Temperature (°C)", -5.0, 30.0, 15.0)
max_temp = st.slider("Max Temperature (°C)", 0.0, 50.0, 25.0)
rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 5.0)
humidity = st.slider("Humidity at 3 PM (%)", 0, 100, 60)
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 20.0)
pressure = st.slider("Pressure (hPa)", 980.0, 1040.0, 1015.0)


if st.button("Predict"):
    input_data = [[min_temp, max_temp, rainfall, humidity, wind_speed, pressure]]
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("☔ **It will rain tomorrow!**")
    else:
        st.success("☀️ **No rain expected tomorrow.**")
    
    st.write(f"Confidence: {proba[prediction] * 100:.1f}%")

    st.subheader("Probability Breakdown")
    prob_df = pd.DataFrame({
        "Outcome": ["No Rain", "Rain"],
        "Probability": proba
    })
    st.bar_chart(prob_df.set_index("Outcome"))

st.markdown("---")
st.subheader("Model Performance")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")

st.subheader("Most Important Features")
importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
st.bar_chart(importance.set_index("Feature"))

# write >>> streamlit run weather_app.csv in terminal to run