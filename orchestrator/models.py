"""SQLAlchemy models for the orchestrator."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, nullable=False)
    harbor_project = Column(String, unique=True, nullable=False, index=True)
    max_submissions = Column(Integer, default=10, nullable=False)
    api_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    submissions = relationship("Submission", back_populates="team", cascade="all, delete-orphan")


class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    image_tag = Column(String, nullable=False)
    status = Column(String, nullable=False, index=True)
    registration_score = Column(Float, nullable=True)
    integration_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    team = relationship("Team", back_populates="submissions")
    logs = relationship("EvaluationLog", back_populates="submission", cascade="all, delete-orphan")


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False, index=True)
    log_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    submission = relationship("Submission", back_populates="logs")


def make_engine(url: str):
    from pathlib import Path

    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(url: str):
    return sessionmaker(bind=make_engine(url), expire_on_commit=False, future=True)
