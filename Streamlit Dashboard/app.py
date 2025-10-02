import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

state_climate = {
    "Gujarat": {"temperature": 25, "humidity": 75, "ph": 7.33},
    "Maharashtra": {"temperature": 27, "humidity": 70, "ph": 6.9},
    "Punjab": {"temperature": 23, "humidity": 80, "ph": 7.8},
    "Tamil Nadu": {"temperature": 29, "humidity": 78, "ph": 6.7},
    "Karnataka": {"temperature": 26, "humidity": 76, "ph": 6.5},
}

st.title("AI Powered Crop Yield Prediction and Optimization")

crop = st.selectbox("Select Crop", crop_encoder.classes_)
season = st.selectbox("Select Season", season_encoder.classes_)
state = st.selectbox("Select State", state_encoder.classes_)

if state in state_climate:
    st.subheader("State Environmental Information")
    st.write(f"**Average Temperature:** {state_climate[state]['temperature']} °C")
    st.write(f"**Average Humidity:** {state_climate[state]['humidity']} %")
    st.write(f"**Average Soil pH:** {state_climate[state]['ph']}")

year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
area = st.number_input("Area (in hectares)", min_value=0.0)
rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0)
fertilizer = st.number_input("Fertilizer Usage (kg/ha)", min_value=0.0)
pesticide = st.number_input("Pesticide Usage (kg/ha)", min_value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Crop": [crop],
        "Crop_Year": [year],
        "Season": [season],
        "State": [state],
        "Area": [area],
        "Production": [0.0],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide]
    })

    input_data["Crop"] = crop_encoder.transform(input_data["Crop"])
    input_data["Season"] = season_encoder.transform(input_data["Season"])
    input_data["State"] = state_encoder.transform(input_data["State"])

    predicted_yield = model.predict(input_data)[0]

    predicted_production = predicted_yield * area

    st.success(f"Predicted Yield: {predicted_yield:.2f} tons/hectare")
    st.info(f"Estimated Production: {predicted_production:.2f} tons")

    st.subheader("📌 Recommendations for Farmers")

    if predicted_yield < 2:
        st.warning("""
        - 🌱 Yield is **low** :
        
          ➡️ Consider Following Things :

          ✅ Improving soil fertility with organic compost & manure.  
          ✅ Using drip irrigation or better water management.  
          ✅ Increasing balanced fertilizer (NPK) usage.  
          ✅ Monitoring soil pH and correcting it if outside optimal range.  
        """)

    elif 2 <= predicted_yield <= 4:
        st.info("""
        - 🌾 Yield is **moderate**. Consider:  
          ✅ Practicing crop rotation to restore soil nutrients.  
          ✅ Using bio-pesticides for pest control.  
          ✅ Applying micronutrients and balanced fertilizers.  
          ✅ Regularly monitoring crop health and rainfall dependency.  
        """)

    else:
        st.success("""
        - 🌟 Yield is **high**. Great job! Consider:  
          ✅ Expanding cultivation area or diversifying crops.  
          ✅ Using precision farming & smart sensors for optimization.  
          ✅ Improving post-harvest storage to avoid losses.  
          ✅ Exploring market linkages and cooperatives for better pricing.  
        """)


