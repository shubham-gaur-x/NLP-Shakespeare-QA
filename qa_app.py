import os
import gdown
import streamlit as st
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Define Google Drive links
MODEL_URL = "https://drive.google.com/uc?id=1kBFDOQ2QgClZ-u2xvR4tilfR63_MXzt6"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"
CONFIG_URL = "https://drive.google.com/uc?id=1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm"

# Define file paths
model_dir = "shakespeare_qa_model"
model_path = os.path.join(model_dir, "model.onnx")
tokenizer_path = os.path.join(model_dir, "tokenizer.json")
config_path = os.path.join(model_dir, "config.json")

# Ensure model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Download files if not present
if not os.path.exists(model_path):
    st.write("Downloading model... This may take a few minutes.")
    gdown.download(MODEL_URL, model_path, quiet=False)

if not os.path.exists(tokenizer_path):
    st.write("Downloading tokenizer... This may take a few seconds.")
    gdown.download(TOKENIZER_URL, tokenizer_path, quiet=False)

if not os.path.exists(config_path):
    st.write("Downloading config... This may take a few seconds.")
    gdown.download(CONFIG_URL, config_path, quiet=False)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load ONNX model
session = ort.InferenceSession(model_path)

# Function to get predictions
def answer_question(question, context):
    # Tokenize input with consistent sequence length
    inputs = tokenizer(
        question,
        context,
        return_tensors="np",
        truncation=True,
        max_length=512,  # Ensure input length matches model expectation
        padding="max_length",  # Pad to the maximum length
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Run ONNX inference
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # Extract start and end logits
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

# Create text inputs for context and question
context = st.text_area("Enter the context (e.g., a passage or scene):", height=200)
question = st.text_input("Enter your question:")

if question and context:
    # Get the answer
    answer = answer_question(question, context)

    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
