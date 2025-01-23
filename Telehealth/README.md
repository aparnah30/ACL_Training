# Multilingual Text Summarizer

## Overview
The **Multilingual Text Summarizer** is a web application built with **Streamlit** and **Hugging Face Transformers** to generate concise summaries of user-provided text. The tool supports multiple languages, allowing users to either input text directly or upload text/PDF files. It uses powerful transformer models like **MT5** for text summarization and **MarianMT** for translation to English.

## Key Features
- **Multilingual Support**: The app can handle text in multiple languages and even translate it to English before summarizing.
- **Text Summarization**: Utilizes the **MT5** model for generating summaries.
- **Translation**: Supports translation from various languages to English using **MarianMT**.
- **ROUGE Scoring**: After generating a summary, the app can calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) to evaluate the quality of the summary.
- **File Upload**: You can upload `.pdf` or `.txt` files, and the app will extract the text and summarize it.
- **Interactive UI**: User-friendly interface with options for direct text input or file upload.

## Resources
This project uses the following resources:
- **Streamlit** for building the web interface.
- **Hugging Face Transformers** for language models (MT5 for summarization, MarianMT for translation).
- **PyMuPDF (fitz)** for reading text from PDF files.
- **LangDetect** for automatic language detection.
- **Rouge-Score** for evaluating summarization quality.

## Installation and Setup

### Clone the repository
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/aparnah30/ACL_Training.git
cd TeleHealth
streamlit run app.py
