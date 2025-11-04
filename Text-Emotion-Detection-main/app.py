import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# ===============================
# 📦 Load Model Safely
# ===============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "text_emotion.pkl")

if os.path.exists(MODEL_PATH):
    pipe_lr = joblib.load(open(MODEL_PATH, "rb"))
else:
    pipe_lr = None
    st.error("❌ Model file not found. Please make sure 'model/text_emotion.pkl' exists.")

# ===============================
# 😀 Emoji Dictionary
# ===============================
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
}

# ===============================
# 🔮 Prediction Functions
# ===============================
def predict_emotions(docx):
    if pipe_lr is not None:
        return pipe_lr.predict([docx])[0]
    return "Model Not Loaded"

def get_prediction_proba(docx):
    if pipe_lr is not None:
        return pipe_lr.predict_proba([docx])
    return [[0.0]]

# ===============================
# 📊 Main Streamlit App
# ===============================
def main():
    st.title("💬 Text Emotion Detection App")
    st.write("This app predicts the emotion from text using a trained ML model.")

    menu = ["Home", "Explore Dataset", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Enter Text to Analyze Emotion")

        with st.form(key='emotion_form'):
            raw_text = st.text_area("Type here:")
            submit_text = st.form_submit_button(label='Predict')

        if submit_text:
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            col1, col2 = st.columns(2)

            with col1:
                st.success("✅ Original Text")
                st.write(raw_text)

                st.success("🎯 Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction} {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("📈 Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                chart = alt.Chart(proba_df_clean).mark_bar().encode(
                    x="Emotions",
                    y="Probability",
                    color="Emotions"
                )
                st.altair_chart(chart, use_container_width=True)

    elif choice == "Explore Dataset":
        st.subheader("📄 Emotion Dataset Preview")
        data_path = os.path.join(BASE_DIR, "data", "emotion_dataset_raw.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.write(df.head(10))
            st.info(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        else:
            st.error("Dataset not found! Please check 'data/emotion_dataset_raw.csv'.")

    else:
        st.subheader("ℹ️ About")
        st.write("""
        **Text Emotion Detection App**
        - Built with Streamlit  
        - Uses a trained NLP model to predict emotions from text  
        - Dataset: emotion_dataset_raw.csv  
        """)

if __name__ == '__main__':
    main()
