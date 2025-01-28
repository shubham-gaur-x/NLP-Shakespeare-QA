# Shakespeare Q&A System

This project is a **Shakespeare Question Answering System** built using **ONNX Runtime**, **Hugging Face Transformers**, and **Streamlit**. The system allows users to ask questions about Shakespeare's works and get relevant answers quickly and efficiently.

---

## Features

- **Fast Inference**: Powered by an ONNX-optimized model for quicker responses.
- **Custom Model**: Fine-tuned question-answering model trained on Shakespeare's works.
- **User-Friendly Interface**: Intuitive web application built with Streamlit.
- **Google Drive Integration**: Downloads required model files automatically from Google Drive.

---

## Table of Contents

- [Demo](#demo)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Setup Instructions

Follow these steps to set up and run the project locally:

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shubham-gaur-x/NLP-Shakespeare-QA.git
   cd NLP-Shakespeare-QA
   ```

2. **Install Dependencies**:
   Install the required Python libraries from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Files**:
   The model and tokenizer files are hosted on Google Drive. The app will automatically download them when you run it for the first time. Ensure you have an internet connection.

   - Model: [model.onnx](https://drive.google.com/file/d/1kBFDOQ2QgClZ-u2xvR4tilfR63_MXzt6/view?usp=sharing)
   - Tokenizer: [tokenizer.json](https://drive.google.com/file/d/1B5WxrV3yLfMBs2ERI_cw7tvDU-ssRR_o/view?usp=sharing)
   - Config: [config.json](https://drive.google.com/file/d/1TWrEmy6jVjCeM1Da5KVgvCmXM8R2MQCm/view?usp=sharing)

4. **Run the App**:
   Launch the Streamlit app:
   ```bash
   streamlit run qa_app.py
   ```

5. Open the app in your browser at `http://localhost:8501`.

---

## Project Structure

```
NLP-Shakespeare-QA/
├── qa_app.py                # Main application file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── shakespeare_text.csv     # Dataset (Shakespeare's works)
├── shakespeare_qa_model/    # Model directory (auto-downloaded files)
│   ├── model.onnx           # ONNX model
│   ├── tokenizer.json       # Tokenizer file
│   └── config.json          # Model configuration
```

---

## How It Works

1. **User Input**:
   - The user enters a question in the app.

2. **Preprocessing**:
   - The question is tokenized using the Hugging Face tokenizer.

3. **ONNX Inference**:
   - The tokenized input is passed to the ONNX model for inference.

4. **Answer Generation**:
   - The model outputs the start and end positions of the answer within the context.
   - The tokenizer decodes the answer tokens into a human-readable string.

5. **Response**:
   - The answer is displayed in the app.

---

## Technologies Used

- **Streamlit**: For building the web application.
- **ONNX Runtime**: For efficient model inference.
- **Hugging Face Transformers**: For tokenization and model training.
- **Python**: The core programming language for this project.

---

## License

This project is licensed under the MIT License. Feel free to use and modify it for your purposes.
