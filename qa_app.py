import os
import gdown
import streamlit as st
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import pandas as pd

# Define Google Drive Links for Model & Tokenizer
MODEL_URL = "https://drive.google.com/uc?id=1kBFDOQ2QgClZ-u2xvR4tilfR63_MXzt6"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

# Define paths for the downloaded files
MODEL_PATH = "shakespeare_qa_model/model.onnx"
TOKENIZER_PATH = "shakespeare_qa_model/tokenizer.json"
CONFIG_PATH = "shakespeare_qa_model/config.json"

# Ensure the directories exist
os.makedirs("shakespeare_qa_model", exist_ok=True)

# Download model and tokenizer if not already available
if not os.path.exists(MODEL_PATH):
    st.write("Downloading ONNX model... Please wait.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(TOKENIZER_PATH):
    st.write("Downloading tokenizer... Please wait.")
    gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)

if not os.path.exists(CONFIG_PATH):
    st.write("Downloading config... Please wait.")
    gdown.download(CONFIG_URL, CONFIG_PATH, quiet=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

# Initialize the ONNX runtime session
session = ort.InferenceSession(MODEL_PATH)

# Load the Shakespeare dataset
# Replace with your dataset if different
df = pd.read_csv("shakespeare_text.csv")  # Ensure the file is uploaded
full_context = " ".join(df["Text"].values)  # Combine all scenes as context

# Function to get predictions
def answer_question(question):
    inputs = tokenizer(
        question,
        full_context,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=17  # Match the model's expected input size
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Run ONNX inference
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # Get the predicted start and end token indices
    start_logits, end_logits = outputs
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    # Decode the tokens between start and end
    answer_tokens = input_ids[0][start_idx : end_idx + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# Streamlit UI
st.title("Shakespeare Q&A System")
st.write("Ask any question about Shakespeare's works, and I'll provide an answer!")

# Input for the user's question
question = st.text_input("Enter your question:")

if question:
    answer = answer_question(question)
    st.subheader("Answer:")
    st.write(answer)
