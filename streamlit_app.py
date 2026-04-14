import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Heart Disease Detection",
    layout="wide",
    page_icon="💖"
)

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
.main { background-color: #fff0f6; }

h1 {
    color: #ff1493;
    text-align: center;
}

.stButton>button {
    background-color: #ff1493;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #ff69b4;
}

section[data-testid="stSidebar"] {
    background-color: #ffe4ec;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "decision_tree_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# ===============================
# FEATURES (IMPORTANT)
# ===============================
FEATURES = [
    "age", "sex", "cp",
    "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ===============================
# TITLE
# ===============================
st.title("💖 Heart Disease Detection System")
st.markdown("### Smart Prediction using Machine Learning")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("🧾 Patient Information")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type(cp)", [0, 1, 2, 3])

thalach = st.sidebar.slider("Max Heart Rate(thalach)", 60, 220, 150)
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
exang = st.sidebar.selectbox("Exercise Angina(exang)", [0, 1])

slope = st.sidebar.selectbox("Slope", [0, 1, 2])
ca = st.sidebar.slider("CA", 0, 3, 0)
thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

# ===============================
# PREPROCESS
# ===============================
def preprocess():
    df = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    return df.reindex(columns=model.feature_names_in_, fill_value=0)

input_data = preprocess()



# ===============================
# PREDICTION
# ===============================
st.markdown("## 🔍 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button("💖 Predict Now"):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        low_risk = proba[0]
        high_risk = proba[1]

        st.write(f"🟢 Low Risk Probability: {low_risk:.2f}")
        st.write(f"🔴 High Risk Probability: {high_risk:.2f}")

        if high_risk > 0.6:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Input Data")
    st.dataframe(input_data)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# DASHBOARD
# ===============================
st.markdown("## 📊 Dashboard")

data_path = BASE_DIR / "data" / "cleaned_data.csv"

try:
    df = pd.read_csv(data_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Age Distribution")
        st.bar_chart(df["age"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cholesterol Trend")
        st.line_chart(df["chol"])
        st.markdown('</div>', unsafe_allow_html=True)

except:
    st.warning("Dataset not found!")

# ===============================
# BATCH PREDICTION
# ===============================
st.markdown("## 📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)

    # Fix missing columns
    for col in FEATURES:
        if col not in df_upload.columns:
            df_upload[col] = 0

    df_upload = df_upload[FEATURES]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Preview:")
    st.dataframe(df_upload.head())

    preds = model.predict(df_upload)
    probs = model.predict_proba(df_upload)[:, 1]

    df_upload["Prediction"] = preds
    df_upload["Risk_Probability"] = probs

    st.write("Results:")
    st.dataframe(df_upload)

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("<center>Made with 💖 | ML Heart Disease App</center>", unsafe_allow_html=True)