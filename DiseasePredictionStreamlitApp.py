import streamlit as st
import requests

# Replace with your actual API URL
API_URL = "https://ddsystem.onrender.com"

st.title("🩺 AI-Powered Self-Learning Medical Assistant")

# ✅ User input for symptoms
user_input = st.text_input("Enter symptoms (separated by spaces):")

if st.button("Predict Disease"):
    if not user_input.strip():
        st.error("Please enter symptoms before predicting.")
    else:
        try:
            response = requests.post(f"{API_URL}/predict", json={"symptoms": user_input})
            if response.status_code == 200:
                result = response.json()
                predicted_disease = result.get("predicted_disease", "Unknown")
                st.success(f"Predicted Disease: **{predicted_disease}**")

                # ✅ Feedback Section
                st.write("🔄 **Help Improve AI**: If the prediction is incorrect, enter the correct disease below.")
                correct_disease = st.text_input("Enter correct disease:")
                
                if st.button("Submit Feedback"):
                    if correct_disease.strip():
                        feedback_response = requests.post(f"{API_URL}/update", json={
                            "symptoms": user_input,
                            "correct_disease": correct_disease.strip()
                        })
                        if feedback_response.status_code == 200:
                            st.success("✅ Thank you! Your feedback has been recorded.")
                        else:
                            st.error(f"❌ Error: {feedback_response.text}")
                    else:
                        st.error("Please enter the correct disease.")

            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")
