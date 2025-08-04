import streamlit as st
from nlp_pipeline import process_text, get_analysis
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="NLTK Text Analyzer")

st.title("📘 NLTK-Powered Text Analytics Web App")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    df = process_text(raw_text)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["📂 Data Explorer", "📊 Analysis Dashboard"])

    if page == "📂 Data Explorer":
        st.header("Cleaned & Processed Data")
        st.dataframe(df)
        st.markdown(f"Total Sentences: `{len(df)}`")

    elif page == "📊 Analysis Dashboard":
        analysis = get_analysis(df)
        st.header("Token Frequency")
        st.bar_chart(analysis['freq_df'].set_index("Token")["Count"].head(10))

        st.header("Top Collocations")
        st.write(", ".join(analysis['collocations'][:10]))

        st.header("Sentiment Score Trend")
        fig, ax = plt.subplots()
        ax.plot(analysis['sentiments'], marker='o')
        ax.set_title("Sentiment Score per Sentence")
        ax.set_ylabel("Sentiment Polarity")
        st.pyplot(fig)
