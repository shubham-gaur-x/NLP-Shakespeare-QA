import os
import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Define Google Drive Links for Model & Tokenizer
MODEL_URL = "https://drive.google.com/uc?id=1T2gM_VXJ1M0QyXycg7XwMPKUzvU2WFsE"
TOKENIZER_URL = "https://drive.google.com/uc?id=1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o"

# Define local model path
model_dir = "shakespeare_qa_model"

# Ensure model files are downloaded
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    gdown.download(MODEL_URL, f"{model_dir}/model.safetensors", quiet=False)
    gdown.download(TOKENIZER_URL, f"{model_dir}/tokenizer.json", quiet=False)

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

# Allow users to select an Act and Scene
act = st.selectbox("Select Act:", df['Act'].unique())
scene = st.selectbox("Select Scene:", df[df['Act'] == act]['Scene'].unique())

# Get selected context
scene_text = df[(df['Act'] == act) & (df['Scene'] == scene)]['Text'].values[0]

# User question input
question = st.text_input("Enter your question:")

if question:
    answer = qa_pipeline(question=question, context=scene_text)['answer']
    
    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
