from sqlalchemy import String
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from database import Base
import time

class Summary(Base):
    __tablename__ = "summaries"
    utc = time.time()
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(String)
    def __repr__(self) -> str:
        return f"summary(id={self.id!r}, text={self.text!r}, summary={self.summary!r})"
