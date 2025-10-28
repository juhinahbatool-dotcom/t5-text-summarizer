
import streamlit as st
from transformers import T5TokenizerFast, T5ForConditionalGeneration, pipeline

# Load fine-tuned model
@st.cache_resource
def load_model():
    model_path = "./model"
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

summarizer = load_model()

st.title("News Article Summarizer (T5)")
st.write("Fine-tuned on the CNN/DailyMail dataset to generate concise summaries.")

article = st.text_area("Enter a news article text here:", height=250)

if st.button("Summarize"):
    if article.strip():
        st.info("Generating summary...")
        summary = summarizer("summarize: " + article, max_length=64, min_length=10, do_sample=False)[0]['summary_text']
        st.success("Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
