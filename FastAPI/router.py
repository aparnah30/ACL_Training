from model import summarize, chunk_text, SummarizeRequest
from fastapi import HTTPException, APIRouter, Depends
from data_models import Summary
from database import get_db
from sqlalchemy.orm import Session
 
router = APIRouter()

@router.post("/summarize")
async def summarize_text(request: SummarizeRequest, db: Session = Depends(get_db)):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        chunks = chunk_text(request.text, request.max_chunk_length) 
        summaries = [summarize(chunk) for chunk in chunks]
        final_summary = ' '.join(summaries)

        db_summary = Summary(text=request.text, summary=final_summary) 
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)

        return {
            "summary": final_summary,
            "original_length": len(request.text.split()),
            "summary_length": len(final_summary.split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
