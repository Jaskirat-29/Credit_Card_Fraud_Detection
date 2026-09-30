import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("random_forest.pkl")
    return preprocessor, model

preprocessor, model = load_models()

st.title("Credit Card Fraud Detection")
st.write("Upload transaction data to detect potentially fraudulent transactions.")

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
  

    df['TransactionDT_hour'] = (df['TransactionDT'] // 3600) % 24
    df['TransactionDT_day'] = df['TransactionDT'] // 86400
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:
        X = preprocessor.transform(df)

        predictions = model.predict(X)

        df["Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df)

        fraud_count = (predictions == 1).sum()

        st.metric("Fraudulent Transactions", int(fraud_count))
        st.metric("Total Transactions", len(predictions))

    except Exception as e:
        st.error(f"Prediction failed: {e}")