import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd

# Load the fine-tuned model and tokenizer
model_dir = "/Users/shubhamgaur/Desktop/NU/Sem3/NLP/Final Exam/shakespeare_qa_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Create the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to answer a question based on the provided context
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Load the Shakespeare text data into a pandas DataFrame
# Replace the path with your actual file path if it's not named 'shakespeare_text.csv'
df = pd.read_csv("shakespeare_text.csv")  # Load your data into a DataFrame

# Streamlit UI elements
st.title("Shakespeare Q&A System")
st.write("Ask any question about Shakespeare's works, and I'll provide an answer!")

# Create a text input for the user to ask questions
question = st.text_input("Enter your question:")

if question:
    # Concatenate all scene texts to form a single large context
    # You can also choose to use a specific play or part of the play if necessary
    context = " ".join(df['Text'].values)  # Concatenate all scene texts for broader context
    
    # Get the answer using the entire text as context
    answer = answer_question(question, context)
    
    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
