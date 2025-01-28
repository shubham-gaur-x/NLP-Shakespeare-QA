import os
import gdown
import streamlit as st
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Paths for downloading model and tokenizer
MODEL_URL = "https://drive.google.com/uc?id=1kBFDOQ2QgClZ-u2xvR4tilfR63_MXzt6"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

MODEL_PATH = "./shakespeare_qa_model/model.onnx"
TOKENIZER_PATH = "./shakespeare_qa_model/tokenizer.json"
CONFIG_PATH = "./shakespeare_qa_model/config.json"

# Ensure the directory exists
os.makedirs("./shakespeare_qa_model", exist_ok=True)

# Download model and tokenizer if not already present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(TOKENIZER_PATH):
    st.write("Downloading tokenizer...")
    gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)

if not os.path.exists(CONFIG_PATH):
    st.write("Downloading config...")
    gdown.download(CONFIG_URL, CONFIG_PATH, quiet=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./shakespeare_qa_model")

# Load the ONNX model
session = ort.InferenceSession(MODEL_PATH)

# Function to answer a question based on context
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="np", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Run ONNX inference
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # Convert the output logits to integer token IDs and decode
    start_logits = outputs[0][0]  # Start logits
    end_logits = outputs[1][0]    # End logits

    # Find the start and end positions
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    # Decode the tokens between start and end
    answer_tokens = input_ids[0][start_idx : end_idx + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# Streamlit UI
st.title("Shakespeare Q&A System")
st.write("Ask any question about Shakespeare's works, and I'll provide an answer!")

# Input from the user
question = st.text_input("Enter your question:")

if question:
    # Example context (replace with actual context from Shakespeare)
    context = "Some Shakespeare context here..."
    answer = answer_question(question, context)

    st.subheader("Answer:")
    st.write(answer)
