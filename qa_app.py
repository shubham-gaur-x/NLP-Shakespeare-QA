import os
import streamlit as st
import pandas as pd
import gdown
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Google Drive links (ensure permissions are set to "Anyone with the link can view")
MODEL_URL = "https://drive.google.com/uc?id=1T2gM_VXJ1M0QyXycg7XwMPKUzvU2WFsE"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

# Define local storage paths
MODEL_DIR = "shakespeare_qa_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download model files if they do not exist
def download_file(url, output_path):
    if not os.path.exists(output_path):
        st.write(f"Downloading {output_path.split('/')[-1]}... This may take a few minutes.")
        gdown.download(url, output_path, quiet=False)
    else:
        st.write(f"{output_path.split('/')[-1]} already exists.")

download_file(MODEL_URL, MODEL_PATH)
download_file(TOKENIZER_URL, TOKENIZER_PATH)
download_file(CONFIG_URL, CONFIG_PATH)

# Load model and tokenizer from local directory
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Caching the QA pipeline for efficiency
@st.cache_resource
def load_pipeline():
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_pipeline()

# Load the Shakespeare dataset
df = pd.read_csv("shakespeare_text.csv")

# Streamlit UI
st.title("Shakespeare Q&A System")
st.write("Ask any question about Shakespeare's works, and I'll provide an answer!")

# User question input
question = st.text_input("Enter your question:")

if question:
    # Select the first 10 records for context
    context = " ".join(df['Text'].values[:10])

    # Get the answer
    result = qa_pipeline(question=question, context=context)

    # Display the answer
    st.subheader("Answer:")
    st.write(result['answer'])
