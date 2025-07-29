import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
 
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
 
@st.cache_data
def load_model():
    """Load the trained model from the file."""
    try:
        model = joblib.load('models/titanic_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please run 'src/train.py' to train and save the model.")
        return None

model = load_model()

# --- DATA PREPROCESSING PIPELINE ---
 
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """Converts user input into a DataFrame that the model can understand."""
    
    # Create a dictionary with user inputs
    input_data = {
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [1 if sex == 'Male' else 0],
        'Embarked_Q': [1 if embarked == 'Queenstown' else 0],
        'Embarked_S': [1 if embarked == 'Southampton' else 0],
    }

    # Convert to DataFrame
    df = pd.DataFrame(input_data)
    
    # Ensure the column order is the same as the one used for training
    # This is a robust way to avoid column order issues.
    # We can get the expected columns from the model itself if it's a pipeline,
    # or define it manually. Let's define it manually for clarity.
    
    expected_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    df = df.reindex(columns=expected_cols)
    
    return df

# --- UI (USER INTERFACE) ---

# Title and Subheader
st.title("🚢 Titanic Survival Predictor")
st.markdown("Did you know? The model predicting your fate was trained on real passenger data from the Titanic disaster. Let's see how you would have fared!")

st.sidebar.header("👤 Enter Your Details:")

# Input fields in the sidebar
pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
sex = st.sidebar.radio("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 29, help="How old were you?")
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.sidebar.slider("Fare (in $)", 0.0, 515.0, 32.0, help="How much did your ticket cost?")
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"], help="Where did you get on the ship?")


# --- PREDICTION LOGIC ---
if st.sidebar.button("🔮 Predict My Fate!", use_container_width=True):
    if model is not None:
        # 1. Preprocess the user input
        processed_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
        
        # 2. Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        
        # Displaying the result in the main area
        st.subheader("📜 The Verdict Is In...")
        
        if prediction == 1:
            st.success("🎉 **You Survived!** 🎉")
            st.balloons()
            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2R4bWZic3F2d3Jpdnh2cDl4ZDJxaTBrMjFrY2F3ZmE4MjV0b3BkayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/yoJC2GnSClbPOkV0eA/giphy.gif", caption="All's well that ends well!")
        else:
            st.error("💔 **You Did Not Survive...** 💔")
            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2VmcW1zdmJvMHZja3AycHF5Zjhna3dqMXh1Zm14czQxbHhrMHFxayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ISOckXUybVfQ4/giphy.gif", caption="Better luck in the next life.")

        # --- Show Prediction Details ---
        st.write("---")
        st.subheader("🧐 Prediction Details")
        
        # Displaying probabilities with a progress bar
        prob_survival = prediction_proba[1] * 100
        prob_demise = prediction_proba[0] * 100
        
        st.write(f"**Probability of Survival:**")
        st.progress(int(prob_survival))
        st.markdown(f"<h5 style='text-align: right; color: green;'>{prob_survival:.2f}%</h5>", unsafe_allow_html=True)
        
        st.write(f"**Probability of Not Surviving:**")
        st.progress(int(prob_demise))
        st.markdown(f"<h5 style='text-align: right; color: red;'>{prob_demise:.2f}%</h5>", unsafe_allow_html=True)
        
        st.info("This prediction is based on a Logistic Regression model trained on historical data.", icon="ℹ️")

    else:
        st.warning("Model is not loaded. Cannot make a prediction.")

# --- FOOTER ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Scikit-learn.")