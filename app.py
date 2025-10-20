import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load trained model and encoders
# -------------------------------
model = joblib.load('bigmart_model.pkl')

# If you also saved your encoders separately, load them here:
# encoder_outlet_id = joblib.load('encoder_outlet_id.pkl')
# encoder_outlet_size = joblib.load('encoder_outlet_size.pkl')
# encoder_outlet_location = joblib.load('encoder_outlet_location.pkl')
# encoder_outlet_type = joblib.load('encoder_outlet_type.pkl')

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Big Mart Sales Predictor", layout="centered")

st.title(" Big Mart Sales Prediction using ML")
st.markdown("This app predicts the **sales** of a product based on various store and item attributes.")

st.subheader("🔹 Enter Product and Outlet Details")

# User inputs
Item_MRP = st.number_input("Item MRP", min_value=0.0, max_value=300.0, step=0.5)

Outlet_Identifier = st.selectbox("Outlet Identifier", 
                                 ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])

Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2020, step=1)

Outlet_Size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])

Outlet_Location_Type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])

Outlet_Type = st.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

# -------------------------------
# Encoding Inputs
# -------------------------------
# Recreate your label encodings manually (must match what you used in training)
outlet_id_map = {'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4,
                 'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9}

outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
outlet_location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
outlet_type_map = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}

# Encode categorical variables
Outlet_Identifier_enc = outlet_id_map[Outlet_Identifier]
Outlet_Size_enc = outlet_size_map[Outlet_Size]
Outlet_Location_Type_enc = outlet_location_map[Outlet_Location_Type]
Outlet_Type_enc = outlet_type_map[Outlet_Type]

# Arrange input in correct order
input_data = np.array([[Item_MRP,
                        Outlet_Identifier_enc,
                        Outlet_Establishment_Year,
                        Outlet_Size_enc,
                        Outlet_Location_Type_enc,
                        Outlet_Type_enc]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales 💰"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales: **{prediction:.2f}**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost")
