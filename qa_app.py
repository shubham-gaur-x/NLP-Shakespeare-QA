import os
import requests
import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Define Google Drive Direct Download Links
MODEL_URL = "https://drive.google.com/uc?id=1T2gM_VXJ1M0QyXycg7XwMPKUzvU2WFsE"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

# Define local model directory
model_dir = "shakespeare_qa_model"
os.makedirs(model_dir, exist_ok=True)

# Function to download a file from a URL
def download_file(url, destination):
    """Download a file from a given URL to the specified destination."""
    if not os.path.exists(destination):
        st.write(f"Downloading {os.path.basename(destination)}... This may take a few minutes.")
        response = requests.get(url, stream=True)
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
    else:
        st.write(f"{os.path.basename(destination)} already exists.")

# Download necessary files
model_path = f"{model_dir}/model.safetensors"
tokenizer_path = f"{model_dir}/tokenizer.json"
config_path = f"{model_dir}/config.json"

download_file(MODEL_URL, model_path)
download_file(TOKENIZER_URL, tokenizer_path)
download_file(CONFIG_URL, config_path)

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
    # Concatenate all text for context (or modify this to focus on specific sections)
    context = " ".join(df['Text'].values)  # Combine all text
    answer = qa_pipeline(question=question, context=context)['answer']

    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
