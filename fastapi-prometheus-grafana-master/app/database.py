"""Database models and setup."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DATABASE_URL, DB_WAIT_SECONDS
from time import sleep


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class Prediction(Base):
    """Database model for predictions."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    image_path: Mapped[str] = mapped_column(String(512))
    model_name: Mapped[str] = mapped_column(String(255))
    inference_ms: Mapped[float] = mapped_column(Float)
    fake_probability: Mapped[float] = mapped_column(Float)
    ground_truth_json: Mapped[str | None] = mapped_column(Text, nullable=True)


# Database engine and session factory
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Initialize the database and ensure it's ready."""
    attempts = max(1, DB_WAIT_SECONDS // 2)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(bind=engine)
            return
        except Exception as exc:
            last_error = exc
            sleep(2)

    raise RuntimeError(f"MySQL is not ready after waiting {DB_WAIT_SECONDS}s: {last_error}")
