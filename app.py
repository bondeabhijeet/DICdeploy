import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="Electric Vehicle Accident Predictor",
    page_icon="🚗",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Accident Predictor", "EV Accident Heatmap"])

with tab1:
    # Title and description
    st.title("🚗 Electric Vehicle Accident Predictor")
    st.markdown("""
    This application predicts the likelihood of casualties in electric vehicle accidents based on various factors.
    The model has been trained on historical accident data from New York State.
    """)

    # Function to validate New York State ZIP codes
    def is_valid_ny_zipcode(zipcode):
        # New York State ZIP code ranges
        ny_ranges = [
            (10001, 14925),  # New York City and surrounding areas
            (12007, 12887),  # Capital Region
            (13001, 13901),  # Central New York
            (14001, 14788),  # Western New York
            (14801, 14925)   # Southern Tier
        ]
        
        try:
            zip_int = int(zipcode)
            return any(lower <= zip_int <= upper for lower, upper in ny_ranges)
        except ValueError:
            return False

    def get_region(zipcode):
        try:
            zip_int = int(zipcode)
            if 10001 <= zip_int <= 14925:
                if 10001 <= zip_int <= 10282:
                    return "Manhattan, NYC"
                elif 10301 <= zip_int <= 10314:
                    return "Staten Island, NYC"
                elif 10451 <= zip_int <= 10475:
                    return "Bronx, NYC"
                elif 11001 <= zip_int <= 11697:
                    return "Queens, NYC"
                elif 11201 <= zip_int <= 11256:
                    return "Brooklyn, NYC"
                else:
                    return "New York City Area"
            elif 12007 <= zip_int <= 12887:
                return "Capital Region"
            elif 13001 <= zip_int <= 13901:
                return "Central New York"
            elif 14001 <= zip_int <= 14788:
                return "Western New York"
            elif 14801 <= zip_int <= 14925:
                return "Southern Tier"
            return "Unknown"
        except ValueError:
            return "Invalid"

    # Load the model
    @st.cache_resource
    def load_model():
        with open('LR_model_f1_0.6133.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temporal Information")
        
        # Date and time input
        date_time = st.date_input("Select Date", datetime.now())
        hour = st.slider("Hour of Day", 0, 23, 12)
        
        # Calculate temporal features
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
        zip_code = st.text_input("ZIP Code", "10001")
        if not zip_code.isdigit() or len(zip_code) != 5:
            st.warning("Please enter a valid 5-digit ZIP code")
        elif not is_valid_ny_zipcode(zip_code):
            st.error("This ZIP code is not in New York State. Please enter a valid NY ZIP code.")
        else:
            region = get_region(zip_code)
            st.success(f"Valid New York State ZIP code.")

    # Predict button
    if st.button("Predict Accident Severity"):
        try:
            # Validate ZIP code before proceeding
            if not is_valid_ny_zipcode(zip_code):
                st.error("Cannot make prediction: Please enter a valid New York State ZIP code.")
                st.stop()
                
            # Create input data
            input_data = pd.DataFrame({
                'Month': [month],
                'Day': [day],
                'Hour': [hour],
                'DayOfWeek': [day_of_week],
                'VEHICLE TYPE CODE 2': [vehicle_type],
                'ZIP CODE': [int(zip_code)],
                'CONTRIBUTING FACTOR VEHICLE 1': [contributing_factor],
                'IsRushHour': [is_rush_hour],
                'IsWeekend': [is_weekend],
                'IsNightTime': [is_night_time]
            })
            
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.header("Prediction Results")
            
            # Create columns for the results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(
                    "Prediction",
                    "Casualty Likely" if prediction[0] == 1 else "No Casualty Likely"
                )
                
            with res_col2:
                st.metric(
                    "Probability",
                    f"{prediction_proba[1]:.2%}"
                )
            
            # Additional information
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

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    ### About the Model
    This model was trained on historical accident data from New York State involving electric vehicles.
    The prediction is based on various factors including:
    - Temporal features (time of day, day of week, etc.)
    - Vehicle type
    - Contributing factors
    - Location (ZIP code)

    The model's performance metric (F1 score) is 0.6133.

    Note: This model is only valid for accidents within New York State ZIP codes.
    Valid NY State ZIP code ranges:
    - New York City and surrounding areas: 10001-14925
    - Capital Region: 12007-12887
    - Central New York: 13001-13901
    - Western New York: 14001-14788
    - Southern Tier: 14801-14925
    """)

with tab2:
    st.title("🗺️ EV Accident Heatmap")
    st.markdown("""
    This heatmap shows the distribution of electric vehicle accidents across New York State.
    The intensity of the color indicates the frequency of accidents in each area.
    """)
    
    # Read and display the heatmap
    try:
        with open('ev_heatmap.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display the heatmap
        components.html(html_content, height=1200)
        
        
    except Exception as e:
        st.error(f"Error loading heatmap: {str(e)}") 