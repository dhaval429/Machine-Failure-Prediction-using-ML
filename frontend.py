import streamlit as st
import pandas as pd
import pickle
st.set_page_config(
    page_title="Failure Prediction Application",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.title("Machine Failure Prediction using ML")

Input1 = st.number_input("Enter the age of equipment (in days)")
Input2 = st.number_input("Enter time since maintenance (in days)")
Input3 = 0
Input4 = st.slider("Enter the value of wear and tear (Standardised value b/w 0 and 1)", 0.0, 1.0)
Input5 = st.number_input("Enter reliability index")
Input6 = st.number_input("Enter the value of usage frequency")
Input7 = st.number_input("Enter the operating temperature (in K)")
Input8 = 1

lst = [[Input1, Input2, Input3, Input4, Input5, Input6, Input7, Input8]]
X_test = pd.DataFrame(lst, columns=['AGE_OF_EQUIPMENT', 'TIME_SINCE_MAINTENANCE', 'CORROSION', 
                                    'WEAR_AND_TEAR', 'RELIABILITY', 'USAGE_FREQUENCY',
                                    'OPERATING_TEMPARATURE', 'OPERATING_KNOWLEDGE'])
# st.write(X_test)
loaded_model = pickle.load(open("model.h5", 'rb'))

if st.button("Find Result!"):
    result = loaded_model.predict(X_test)
    if result==0:
        st.subheader('The machine will not fail')
    else:
        st.subheader('The machine requires maintenance')


