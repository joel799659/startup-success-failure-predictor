import streamlit as st
import pickle
import pandas as pd

# 🔧 Streamlit config — must be the first Streamlit command
st.set_page_config(page_title="Startup Success Predictor", page_icon="🚀")

# ✅ Load your trained pipeline model
@st.cache_data
def load_model():
    with open("startup.pkl", "rb") as f:
        return pickle.load(f)

# 📂 Load your dataset to populate dropdowns
@st.cache_data
def load_data():
    return pd.read_csv("big_startup_secsees_dataset.csv")  # your dataset file

# 🧠 Prediction function
def predict_startup_success(pipeline_model, funding_total_usd, funding_rounds, startup_age, funding_duration,
                            category_list, country_code):
    import pandas as pd

    # Create input sample as a DataFrame
    input_data = pd.DataFrame([{
        'funding_total_usd': funding_total_usd,
        'funding_rounds': funding_rounds,
        'startup_age': startup_age,
        'funding_duration': funding_duration,
        'category_list': category_list,
        'country_code': country_code
    }])

    # Get probability of success from model
    probability = pipeline_model.predict_proba(input_data)[0][1]
    prediction = int(probability >= 0.8)  # Apply custom threshold
    return prediction, probability

# 🚀 Load resources
pipeline = load_model()
df = load_data()

# 🎯 App title and intro
st.title("🚀 Startup Success Predictor")
st.markdown("Predict the success of a startup based on funding and category details.")

# 📝 Input Form
with st.form("prediction_form"):
    st.subheader("Enter Startup Details")

    funding_total_usd = st.number_input("Funding Total (USD)", min_value=0.0, step=1000.0)
    funding_rounds = st.number_input("Number of Funding Rounds", min_value=0)
    startup_age = st.number_input("Startup Age (in years)", min_value=0)
    funding_duration = st.number_input("Funding Duration (months)", min_value=0)

    category_options = sorted(df['category_list'].dropna().unique())
    category_list = st.selectbox("Category", category_options)

    country_options = sorted(df['country_code'].dropna().unique())
    country_code = st.selectbox("Country Code", country_options)

    submit = st.form_submit_button("Predict")

# 🔍 Prediction Result
if submit:
    prediction, probability = predict_startup_success(
        pipeline,
        funding_total_usd,
        funding_rounds,
        startup_age,
        funding_duration,
        category_list,
        country_code
    )

    if prediction == 1:
        st.success(f"✅ The startup is likely to succeed! Probability: {probability:.2%}")
    else:
        st.error(f"❌ The startup is likely to fail. Probability of success: {probability:.2%}")
