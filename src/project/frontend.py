"""Streamlit frontend for the ML API."""

import streamlit as st
import requests

st.set_page_config(page_title="Finance Sentiment classification", layout="centered")

st.title("Finance Sentiment classification")

API_BASE_URL = "http://localhost:8000"

# Health Check Section
st.header("Health Check")
if st.button("Check API Health"):
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            st.success("✅ API is healthy!")
            st.json(response.json())
        else:
            st.error(f"API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Training Section
st.header("Train Model")
col1, col2 = st.columns(2)

with col1:
    epochs = st.number_input("Epochs", min_value=1, value=3)
with col2:
    batch_size = st.number_input("Batch Size", min_value=1, value=32)

lr = st.number_input("Learning Rate", min_value=0.0001, value=0.001, step=0.0001)

if st.button("Start Training"):
    with st.spinner("Training in progress..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/train", json={"epochs": epochs, "batch_size": batch_size, "lr": lr}
            )
            if response.status_code == 200:
                st.success("Training completed!")
                st.json(response.json())
            else:
                st.error(f"Training failed with status code: {response.status_code}")
                st.json(response.json())
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.info("💡 Make sure the API is running on http://localhost:8000")
