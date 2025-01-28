import os
import gdown
import streamlit as st
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Google Drive URLs
MODEL_URL = "https://drive.google.com/uc?id=1kBFDOQ2QgClZ-u2xvR4tilfR63_MXzt6"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

# Define paths
MODEL_DIR = "shakespeare_qa_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download the ONNX model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.write("Downloading ONNX model... This may take a few minutes.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Download the tokenizer if it doesn't exist
if not os.path.exists(TOKENIZER_PATH):
    st.write("Downloading tokenizer...")
    gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)

# Download the config if it doesn't exist
if not os.path.exists(CONFIG_PATH):
    st.write("Downloading config...")
    gdown.download(CONFIG_URL, CONFIG_PATH, quiet=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load ONNX Model
session = ort.InferenceSession(MODEL_PATH)

# Function to get predictions
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Run ONNX inference
    output = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    return tokenizer.decode(output[0][0])

# Streamlit UI
st.title("Shakespeare Q&A System")
st.write("Ask any question about Shakespeare's works!")

question = st.text_input("Enter your question:")
if question:
    context = "Some Shakespeare context here..."  # Replace with actual text
    answer = answer_question(question, context)
    st.subheader("Answer:")
    st.write(answer)
