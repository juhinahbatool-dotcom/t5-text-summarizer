import streamlit as st
from transformers import T5TokenizerFast, T5ForConditionalGeneration, pipeline
import os, zipfile, gdown

# ------------- CONFIG -------------
MODEL_DIR = "./model"
DRIVE_FILE_ID = "1levZURbnocK_uN64jB3yXTaOAYktoKfk"   
ZIP_PATH = "https://drive.google.com/file/d/1levZURbnocK_uN64jB3yXTaOAYktoKfk/view?usp=sharing"
# ----------------------------------

# Load fine-tuned model
@st.cache_resource
def load_model_from_drive():
    # Download from Google Drive 
    if not os.path.exists(MODEL_DIR):
        st.info("Downloading fine-tuned model from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        st.success("Model downloaded and extracted successfully!")

    # Load model
    st.info("Loading model...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

st.title("News Article Summarizer (T5)")
st.write("Fine-tuned on the CNN/DailyMail dataset — loaded from Google Drive.")

summarizer = load_model_from_drive()
article = st.text_area("Enter a news article to summarize:", height=250)

if st.button("Summarize"):
    if article.strip():
        st.info("Generating summary...")
        summary = summarizer(
            "summarize: " + article,
            max_length=64,
            min_length=10,
            do_sample=False,
            num_beams=4
        )[0]["summary_text"]
        st.success("### Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")
