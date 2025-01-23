import streamlit as st
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  
import os
import re
from langdetect import detect
from rouge_score import rouge_scorer

st.set_page_config(page_title="Multilingual Text Summarizer", layout="wide")

@st.cache_resource
def load_model():
    model_directory = "ctu-aic/mt5-base-multilingual-summarization-multilarge-cs"
    model = MT5ForConditionalGeneration.from_pretrained(model_directory)
    tokenizer = MT5Tokenizer.from_pretrained(model_directory, legacy = False)
    return model, tokenizer

model, tokenizer = load_model()

@st.cache_resource
def load_translation_models():
    # # Load translation models
    # translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M",src_lang="de", tgt_lang="en")
    translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")

    return translation_model, translation_tokenizer

translation_model, translation_tokenizer = load_translation_models()

def translate_text(text, src_lang):
    src_lang = src_lang.lower()
    if src_lang == "zh-cn":
        src_lang = "zh"
    translation_input = translation_tokenizer.prepare_seq2seq_batch([text], src_lang=src_lang, tgt_lang="en", return_tensors="pt")
    translated_ids = translation_model.generate(**translation_input)
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def preprocess_text(text):
    # Remove special characters and extra whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def summarize_text(text, prompts):
    cleaned_text = preprocess_text(text)
    combined_text = f"summarize: {cleaned_text}"
    if prompts:
        combined_text += " " + " ".join(prompts)
    
    tokenized_text = tokenizer.encode(combined_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    summary_ids = model.generate(tokenized_text, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def detect_language(text):
    lang = detect(text)
    return lang

# Add function to calculate ROUGE scores
def calculate_rouge_score(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

# App layout
st.title("Multilingual Text Summarizer")
st.write("**Welcome to the multilingual text summarizer!** Enter your text directly or upload a text/PDF file below, and let's create a concise summary together. 🧠")
st.write("**Using Transformer Model: T5**")

# Sidebar input method selection
st.sidebar.write("### Input Method")
input_method = st.sidebar.radio("Choose input method:", ("Direct Text Input", "Upload File (PDF, TXT)"))

if input_method == "Direct Text Input":
    # Text input
    user_input = st.text_area("Enter your text here:", height=200)

    if user_input:
        file_text = user_input
    else:
        file_text = None

else:
    # File upload
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            file_text = read_pdf(uploaded_file)
        elif file_extension == ".txt":
            file_text = read_txt(uploaded_file)
        else:
            file_text = None
            st.error("Unsupported file type. Please upload a PDF, TXT file.")
    else:
        file_text = None

if file_text:
    if input_method == "Upload File (PDF, TXT)":
        st.write("**File/Text content:**")
        st.text_area("File/Text content", value=file_text, height=200)

    # Detect language
    detected_language = detect_language(file_text)
    st.write(f"**Detected Language:** {detected_language.capitalize()}")

    # Translation option
    if detected_language != "en":
        translate_option = st.checkbox("Translate to English")
        if translate_option:
            file_text = translate_text(file_text, detected_language)
            st.write("**Translated Text:**")
            st.text_area("Translated Text", value=file_text, height=200)
            detected_language = "en"

    # Chat-like prompt system
    if "prompts" not in st.session_state:
        st.session_state.prompts = []

    # st.write("### Refine your summary:")
    # prompt = st.text_input("Enter a prompt to refine the summary, e.g., 'focus on key points'")

    # if st.button("Add Prompt"):
    #     if prompt:
    #         st.session_state.prompts.append(prompt)
    #         st.success(f"Prompt added: {prompt}")
    #     else:
    #         st.error("Please enter a valid prompt.")

    # # Display current prompts
    # if st.session_state.prompts:
    #     st.write("#### Current Prompts:")
    #     for i, p in enumerate(st.session_state.prompts):
    #         st.write(f"{i+1}. {p}")

    # Summary button
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                # Generate summary
                summary = summarize_text(file_text, st.session_state.prompts)
                st.subheader("Summary")
                st.write(summary)
                
                # If reference summary is provided, calculate ROUGE score
                if summary:
                    rouge_scores = calculate_rouge_score(summary, file_text)
                    st.write("**ROUGE Scores:**")
                    st.write(f"ROUGE-1: {rouge_scores['rouge1']}")
                    st.write(f"ROUGE-2: {rouge_scores['rouge2']}")
                    st.write(f"ROUGE-L: {rouge_scores['rougeL']}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.write("Please enter some text or upload a file to get started.")
