import streamlit as st
import pandas as pd
import pickle
st.set_page_config(
    page_title="Failure Prediction Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("BTP")

st.write("Machine Failure Prediction using ML")

Input1 = st.number_input("Enter the age of equipment")
Input2 = st.number_input("Enter time since maintenance")
Input3 = st.number_input("Enter the value of corrosion")
Input4 = st.number_input("Enter the value of wear and tear")
Input5 = st.number_input("Enter reliability index")
Input6 = st.number_input("Enter the value of usage frequency")
Input7 = st.number_input("Enter the operating frequency")
Input8 = st.number_input("Enter the operating knowledge")

lst = [[Input1, Input2, Input3, Input4, Input5, Input6, Input7, Input8]]
X_test = pd.DataFrame(lst, columns=['AGE_OF_EQUIPMENT','TIME_SINCE_MAINTENANCE','CORROSION','WEAR_AND_TEAR','RELIABILITY','USAGE_FREQUENCY','OPERATING_TEMPARATURE','OPERATING_KNOWLEDGE'])
st.write(X_test)
loaded_model = pickle.load(open("model.h5", 'rb'))
result = loaded_model.predict(X_test)

st.write(result)


