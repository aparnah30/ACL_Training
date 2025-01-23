# Research Paper Summarizer using T5 (Fine-Tuning)

This project involves fine-tuning a pre-trained **T5 (Text-to-Text Transfer Transformer)** model for the task of summarizing research papers. The model is fine-tuned using a dataset of research papers and their corresponding abstracts (summaries). The pipeline is built using **PyTorch Lightning** for efficient model training and **Hugging Face Transformers** for leveraging the pre-trained T5 model.

## Project Overview

The goal of this project is to build an automated research paper summarizer using the **T5-base** model. The model is fine-tuned on a dataset containing research paper articles and their corresponding abstracts, and is trained to generate summaries for unseen papers.

### Key Steps:
1. **Data Preprocessing**: A dataset of research papers and their abstracts is preprocessed and split into training, validation, and test sets.
2. **Model Setup**: The T5 model is fine-tuned using the processed dataset.
3. **Evaluation**: The trained model is evaluated using various metrics to measure its summarization performance.

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- PyTorch Lightning 1.5.0+
- Hugging Face Transformers 4.0.0+
- scikit-learn
- pandas
- numpy
- tqdm

You can install all dependencies by running:

```bash
pip install -r requirements.txt
