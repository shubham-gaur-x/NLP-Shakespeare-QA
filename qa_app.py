import os
import streamlit as st
import pandas as pd
import gdown
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig

# Google Drive links for model components
MODEL_URL = "https://drive.google.com/uc?id=1T2gM_VXJ1M0QyXycg7XwMPKUzvU2WFsE"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

# Define paths
MODEL_DIR = "shakespeare_qa_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download model components if not present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... This may take a few minutes.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(TOKENIZER_PATH):
    st.write("Downloading tokenizer...")
    gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)

if not os.path.exists(CONFIG_PATH):
    st.write("Downloading config...")
    gdown.download(CONFIG_URL, CONFIG_PATH, quiet=False)

# Load the model and tokenizer with explicit config
config = AutoConfig.from_pretrained(CONFIG_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, config=config)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Caching the QA pipeline to avoid reloading on every request
@st.cache_resource
def load_pipeline():
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_pipeline()

# Load the Shakespeare text data
df = pd.read_csv("shakespeare_text.csv")

# Streamlit UI
st.title("Shakespeare Q&A Bot")
st.write("Ask any question about Shakespeare's works, and I'll provide an answer!")

# User input
question = st.text_input("Enter your question:")

if question:
    # Limit context size to avoid performance issues
    context = " ".join(df['Text'].values[:10])  # Using only first 10 scenes

    # Get answer
    result = qa_pipeline(question=question, context=context)

    # Display answer
    st.subheader("Answer:")
    st.write(result['answer'])
