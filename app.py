import streamlit as st
import pickle
import pandas as pd
import time

emotion_numbers = pickle.load(open("emotion_mapping.pkl", "rb"))
emotion_map = {v: k for k, v in emotion_numbers.items()}

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# ==========================================================
# EMOTION MAPPING
# ==========================================================

emotion_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

emoji = {
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😄"
}

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#eef2ff,#dbeafe,#f8fafc);
}

.main-title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#1e3a8a;
}

.subtitle{
font-size:18px;
text-align:center;
color:#374151;
margin-bottom:20px;
}

.result-box{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0px 5px 25px rgba(0,0,0,.15);
margin-top:20px;
margin-bottom:20px;
}

...

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📘 Project Information")

st.sidebar.success("Emotion Detection using NLP")

st.sidebar.markdown("---")

st.sidebar.write("### Model")

st.sidebar.write("✅ Logistic Regression")

st.sidebar.write("### Vectorizer")

st.sidebar.write("✅ TF-IDF")

st.sidebar.write("### Accuracy")

st.sidebar.success("86 %")

st.sidebar.markdown("---")

st.sidebar.write("### Supported Emotions")

for e in emotion_map.values():
    st.sidebar.write(f"{emoji[e]} {e.title()}")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Built with

- Python
- Scikit-Learn
- Streamlit
- TF-IDF
"""
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
"<div class='main-title'>😊 Emotion Detection System</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>Predict Human Emotions using Machine Learning</div>",
unsafe_allow_html=True
)

st.write("")

# ==========================================================
# INPUT
# ==========================================================

user_input = st.text_area(
"✍ Enter your text",
height=180,
placeholder="Example: Today is the happiest day of my life."
)

col1,col2 = st.columns([1,1])

with col1:
    predict = st.button(
        "🔍 Predict Emotion",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear:
    st.rerun()

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    if user_input.strip()=="":

        st.warning("Please enter some text.")

    else:

        with st.spinner("Analyzing Emotion..."):

            time.sleep(1)

            vector = vectorizer.transform([user_input])

            pred = int(model.predict(vector)[0])

            probabilities = model.predict_proba(vector)[0]

            confidence = probabilities.max()*100

            emotion = emotion_map[pred]

        st.success("Prediction Completed Successfully!")

        st.markdown("<div class='result-box'>",unsafe_allow_html=True)

        st.markdown(
            f"""
            <h1 style='text-align:center;'>
            {emoji[emotion]}
            </h1>

            <h2 style='text-align:center;color:#2563eb;'>
            {emotion.upper()}
            </h2>

            <h3 style='text-align:center;'>
            Confidence : {confidence:.2f}%
            </h3>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>",unsafe_allow_html=True)

        st.subheader("📊 Prediction Probabilities")

        class_names = [
            emotion_map[i]
            for i in range(len(probabilities))
        ]

        prob_df = pd.DataFrame({

            "Emotion":class_names,

            "Probability":probabilities

        })

        prob_df = prob_df.sort_values(
            by="Probability",
            ascending=False
        )

        for _,row in prob_df.iterrows():

            st.write(
                f"**{emoji[row['Emotion']]} {row['Emotion'].title()}**"
            )

            st.progress(float(row["Probability"]))

            st.write(
                f"{row['Probability']*100:.2f}%"
            )

            st.write("")
            
    # ==========================================================
# BAR CHART
    # ==========================================================

    st.subheader("📈 Emotion Probability Chart")

    chart_df = prob_df.copy()
    chart_df["Probability"] = chart_df["Probability"] * 100
    chart_df = chart_df.set_index("Emotion")

    st.bar_chart(chart_df)

    # ==========================================================
    # TOP 3 PREDICTIONS
    # ==========================================================

    st.subheader("🏆 Top Predictions")

    top3 = prob_df.head(3)

    rank = 1

    for _, row in top3.iterrows():

        st.write(
            f"{rank}. {emoji[row['Emotion']]} "
            f"**{row['Emotion'].title()}** "
            f"({row['Probability']*100:.2f}%)"
        )

        rank += 1

    # ==========================================================
    # EXAMPLES
    # ==========================================================

    st.markdown("---")

    st.subheader("💡 Try These Example Sentences")

    example1, example2, example3 = st.columns(3)

    with example1:

        st.info("I am feeling amazing today.")

        st.info("I finally achieved my dream.")

        st.info("I won the competition.")

    with example2:

        st.info("I miss my family.")

        st.info("I am scared of tomorrow.")

        st.info("I feel very lonely.")

    with example3:

        st.info("I love spending time with you.")

        st.info("Wow! I never expected this.")

        st.info("Why did this happen to me?")

    # ==========================================================
    # ABOUT PROJECT
    # ==========================================================

    st.markdown("---")

    st.subheader("📖 About This Project")

    st.write(
    """
    This application predicts human emotions from text using
    Natural Language Processing.

    ### Workflow

    1. User enters text.
    2. Text is converted into TF-IDF features.
    3. Logistic Regression predicts the emotion.
    4. Prediction probabilities are displayed.

    Supported emotions:

    - 😢 Sadness
    - 😠 Anger
    - ❤️ Love
    - 😲 Surprise
    - 😨 Fear
    - 😄 Joy
    """
    )

    # ==========================================================
    # FOOTER
    # ==========================================================

    st.markdown("---")

    st.markdown(
    """
    <div class='footer'>

    <h4>Emotion Detection using NLP</h4>

    Built with ❤️ using Streamlit, Scikit-Learn and Python

    TF-IDF + Logistic Regression

    </div>
    """,
    unsafe_allow_html=True
    )