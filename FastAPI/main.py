from fastapi import FastAPI
from router import router
from data_models import Base
from database import engine
from fastapi.staticfiles import StaticFiles 

app = FastAPI(
    title="Text Summarizer API",
    description="API for text summarization using fine-tuned T5 model",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static") 

app.include_router(router)
Base.metadata.create_all(bind=engine)
