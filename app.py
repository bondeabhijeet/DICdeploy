import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import streamlit.components.v1 as components

# Giving page a logo title and defining a layout
st.set_page_config(
    page_title="Electric Vehicle Accident Predictor",
    page_icon="🚗",
    layout="wide"
)

# Adding custom CSS
st.markdown("""
    <style>
    .main {
        padding: 3rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Making tabs
tab1, tab2 = st.tabs(["Accident Predictor", "EV's Accident Heatmap"])

with tab1:
    # Title and description for the application
    st.title("🚗 Electric Vehicle Accident Predictor")
    st.markdown("""This application predicts the probability of casualties in electric vehicle accidents based on various factors mentioned in the lower end of this page.
    The model has been trained on dataset of accidents from New York City.""")

    # Function to validate New York City ZIP codes
    def is_valid_ny_zipcode(zipcode):
        # New York City ZIP code ranges
        NewYorkRanges = [
            (10001, 14925),  # New York City and surrounding areas
            (14925, 15000)  # Capital Region
        ]
        
        try:
            ZipCodeInt = int(zipcode)
            return any(lower <= ZipCodeInt <= upper for lower, upper in NewYorkRanges)
        except ValueError:
            return False

    def GetRegion(zipcode):
        try:
            ZipCodeInt = int(zipcode)
            if 10001 <= ZipCodeInt <= 14925:
                if 10001 <= ZipCodeInt <= 10282:
                    return "Manhattan, NYC"
                elif 10301 <= ZipCodeInt <= 10314:
                    return "Staten Island, NYC"
                elif 10451 <= ZipCodeInt <= 10475:
                    return "Bronx, NYC"
                elif 11001 <= ZipCodeInt <= 11697:
                    return "Queens, NYC"
                elif 11201 <= ZipCodeInt <= 11256:
                    return "Brooklyn, NYC"
                else:
                    return "New York City Area"
            elif 12007 <= ZipCodeInt <= 12887:
                return "Capital Region"
            elif 13001 <= ZipCodeInt <= 13901:
                return "Central New York"
            elif 14001 <= ZipCodeInt <= 14788:
                return "Western New York"
            elif 14801 <= ZipCodeInt <= 14925:
                return "Southern Tier"
            return "Unknown"
        except ValueError:
            return "Invalid"

    # Loading the model
    @st.cache_resource
    def LoadingModel():
        with open('LR_model_f1_0.6133.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    try:
        model = LoadingModel()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Createing the columns for the layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Information")
        
        # For input
        date_time = st.date_input("Select Date", datetime.now())
        hour = st.slider("Hour of Day", 0, 23, 12)
        
        # Calculate time related features
        month = date_time.month
        day = date_time.day
        day_of_week = date_time.weekday()  # 0=Monday, 6=Sunday
        is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night_time = 1 if (hour >= 22) or (hour <= 5) else 0

    with col2:
        st.subheader("Vehicle and Location Information")
        
        # Vehicle type selection
        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["sedan", "suv", "bus", "bicycle", "truck", "van", "motorcycle"]
        )
        
        # Contributing factor selection
        contributing_factor = st.selectbox(
            "Contributing Factor",
            [
                "driver inattention/distraction",
                "failure to yield right-of-way",
                "following too closely",
                "unsafe speed",
                "unsafe lane changing",
                "backing unsafely",
                "other"
            ]
        )
        
        # ZIP code input
        ZipCodeText = st.text_input("Zip Code", "10001")
        if not ZipCodeText.isdigit() or len(ZipCodeText) != 5:
            st.warning("Please enter a valid 5-digit ZIP code")
        elif not is_valid_ny_zipcode(ZipCodeText):
            st.error("This ZIP code is not in New York City. Please enter a valid NY ZIP code.")
        else:
            region = GetRegion(ZipCodeText)
            st.success(f"Valid New York City ZIP code.")

    # Making the prediction button
    if st.button("Predict Accident Severity"):
        try:
            # Check if the zip code is in new york
            if not is_valid_ny_zipcode(ZipCodeText):
                st.error("Cannot make prediction: Please enter a valid New York City ZIP code.")
                st.stop()
                
            # Creating input data
            input_data = pd.DataFrame({
                'Month': [month],
                'Day': [day],
                'Hour': [hour],
                'DayOfWeek': [day_of_week],
                'VEHICLE TYPE CODE 2': [vehicle_type],
                'ZIP CODE': [int(ZipCodeText)],
                'CONTRIBUTING FACTOR VEHICLE 1': [contributing_factor],
                'IsRushHour': [is_rush_hour],
                'IsWeekend': [is_weekend],
                'IsNightTime': [is_night_time]
            })
            
            # makeing predictions
            prediction = model.predict(input_data)
            PredictionProbability = model.predict_proba(input_data)[0]
            
            # Displaying results
            st.header("Prediction Results")
            
            # Creating new columns for the results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(
                    "Prediction",
                    "Casualty Likely" if prediction[0] == 1 else "No Casualty Likely"
                )
                
            with res_col2:
                st.metric(
                    "Probability",
                    f"{PredictionProbability[1]:.2%}"
                )
            
            # Calculating risk related information
            st.markdown("### Risk Factors")
            risk_factors = []
            if is_rush_hour:
                risk_factors.append("- Accident occurs during rush hour")
            if is_night_time:
                risk_factors.append("- Accident occurs during night time")
            if contributing_factor in ["driver inattention/distraction", "unsafe speed"]:
                risk_factors.append(f"- High-risk contributing factor: {contributing_factor}")
                
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No significant risk factors identified.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Adding information related to models
    st.markdown("---")
    st.markdown("""
    ### About the Model
    This model was trained on old accident data from New York City involving electric vehicles.
    The prediction is based on various factors including:
    - time of day, day of week, etc.
    - Vehicle type
    - Contributing factors
    - ZIP code

    The model's F1 score is 0.6133.

    Note: This model is only valid for accidents within New York City ZIP codes.
    Valid NY City ZIP code ranges:
    - New York City and surrounding areas: 10001-14925""")

with tab2:
    st.title("🗺️ EV Accident Heatmap")
    st.markdown("""
    This heatmap shows the distribution of electric vehicle accidents across New York City.
    The intensity of the color indicates the frequency of accidents in each area. Here red means that there are more accidents, and green means less number of accidents.
    """)
    
    # Reading and display the heatmap
    try:
        with open('ev_heatmap.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Displaying the heatmap
        components.html(html_content, height=1200)
        
        
    except Exception as e:
        st.error(f"Error loading heatmap: {str(e)}") 