import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, label_encoder

model = joblib.load(r'Model/RandomForestClassifier.pkl')
le = joblib.load(r'Model/label_encoder_feature.pkl')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
option_sex  = ['Male','Female']
option_Educational_level       = ['Above high school', 'Junior high school','Elementary school', 'High school', 'Unknown', 'Illiterate','Writing & reading']
option_vechile_driver_relation = ['Employee', 'Unknown', 'Owner', 'Other']
options_driver_exp             = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
option_owner_of_vehicle        = ['Owner', 'Governmental','Organization', 'Other']
option_vehicle_service_year     = ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown','Below 1yr']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

options_lanes  = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
       
option_day_time =  ['Evening', 'Night', 'Afternoon', 'Morning']
       
       

features = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver','Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
       'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians','Number_of_vehicles_involved'
       'Casualty_severity', 'Cause_of_accident','Time of Day']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)


def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        #hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        Day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
     Age_band_of_driver = st.selectbox("Select Driver Age: ", options=options_age)    
     Sex_of_driver = st.selectbox("Select Driver Gender: ", options=option_sex)
     Educational_level = st.selectbox("Select Educational level: ", options=option_Educational_level) 
        Vehicle_driver_relation = st.selectbox("Select Relationship with driver: ", options=option_vechile_driver_relation)
        Driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)        
        Type_of_vehicle = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        Owner_of_vehicle = st.selectbox("Select Vehicle Owner: ", options=option_owner_of_vehicle)  
        Service_year_of_vehicle = st.selectbox("Select Vehicle Service Year: ", options=option_vehicle_service_year) 
        Area_accident_occured = st.selectbox("Select Accident Area: ", options=options_acc_area)   
        Lanes_or_Medians = st.selectbox("Select Lanes: ", options=options_lanes)       
        Number_of_vehicles_involved = st.slider("Pickup Hour: ", 1, 7, value=0, format="%d")
        Age_band_of_casualty = st.selectbox("Select Casuality Age: ", options=options_age)        
        Casualty_severity = st.slider("Hour of Accident: ", 1, 8, value=0, format="%d")
        Cause_of_accident = st.selectbox("Select Accident Cause: ", options=options_cause) 
        Time_of_Day = st.selectbox("Select Time of the day: ", options=option_day_time)        
       
        submit = st.form_submit_button("Predict")


    if submit:
      Day_of_week = label_encoder(Day_of_week,le)
		  Age_band_of_driver = label_encoder(Age_band_of_driver,le)
		  Sex_of_driver = label_encoder(Sex_of_driver,le)
		  Educational_level = label_encoder(Educational_level,le)
		  Vehicle_driver_relation = label_encoder(Vehicle_driver_relation,le)
		  Driving_experience = label_encoder(Driving_experience,le)
		  Type_of_vehicle = label_encoder(Type_of_vehicle,le)
		  Owner_of_vehicle = label_encoder(Owner_of_vehicle,le)
		  Service_year_of_vehicle = label_encoder(Service_year_of_vehicle,le)
		  Area_accident_occured = label_encoder(Area_accident_occured,le)
		  Lanes_or_Medians = label_encoder(Lanes_or_Medians,le)
		  Age_band_of_casualty = label_encoder(Age_band_of_casualty,le)
		  Cause_of_accident = label_encoder(Cause_of_accident,le)
		  Time_of_Day = label_encoder(Time_of_Day,le)
		

       data = np.array([Day_of_week, Age_band_of_driver, Sex_of_driver,Educational_level, Vehicle_driver_relation, Driving_experience,
       Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,Area_accident_occured, Lanes_or_Medians,Number_of_vehicles_involved,
       Age_band_of_casualty,Casualty_severity, Cause_of_accident,Time_of_Day]).reshape(1,-1)
                            
        
                            
        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()
