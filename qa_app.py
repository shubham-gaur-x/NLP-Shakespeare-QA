import os
import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Define Google Drive Direct Download Links (Update FILE_IDs)
MODEL_URL = "https://drive.google.com/uc?id=1T2gM_VXJ1M0QyXycg7XwMPKUzvU2WFsE"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"

# Define local model path
model_dir = "shakespeare_qa_model"

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Check if model & tokenizer exist; if not, download them
model_path = f"{model_dir}/model.safetensors"
tokenizer_path = f"{model_dir}/tokenizer.json"

if not os.path.exists(model_path):
    st.write("Downloading model... This may take a few minutes.")
    gdown.download(MODEL_URL, model_path, quiet=False)

if not os.path.exists(tokenizer_path):
    st.write("Downloading tokenizer... This may take a few minutes.")
    gdown.download(TOKENIZER_URL, tokenizer_path, quiet=False)

# Load Model and Tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Create the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load Shakespeare text data
df = pd.read_csv("shakespeare_text.csv")

# Streamlit UI
st.title("Shakespeare Q&A System")
st.write("Ask any question about Shakespeare's works, and I'll provide an answer!")

# User question input
question = st.text_input("Enter your question:")

if question:
    # Use the first scene text as context (Replace this with a smarter context selection logic)
    context = df.iloc[0]['Text']

    # Get the answer
    answer = qa_pipeline(question=question, context=context)['answer']

    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
