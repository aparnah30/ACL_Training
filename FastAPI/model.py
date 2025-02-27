import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
import nltk
from pydantic import BaseModel
nltk.download('punkt')

TEXT_LEN = 512
HEADLINE_LEN = 64
DEVICE = "cpu"

class SummaryModel(pl.LightningModule):
    def __init__(self):
        super(SummaryModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs.loss, outputs.logits

    def configure_optimizers(self): 
        optimizer = AdamW(self.model.parameters(), lr=0.0001)
        return optimizer

print("Loading model and tokenizer...")
try:
    model = SummaryModel()      
    state_dict = torch.load("models/model_weights2/pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

class SummarizeRequest(BaseModel):
    text: str
    max_chunk_length: int = 100

def chunk_text(text: str, max_length: int) -> list:
    sentences = nltk.sent_tokenize(text)
    current_chunk = []
    current_length = 0
    chunks = []

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize(text: str) -> str:
    text = "summarize: " + text
    
    inputs = tokenizer(
        text,
        max_length=TEXT_LEN,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        summarized_ids = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=HEADLINE_LEN,
            early_stopping=True
        )

    return tokenizer.decode(summarized_ids[0], skip_special_tokens=True)