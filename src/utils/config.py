import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Config:

    # -----------------------------
    # Paths
    # -----------------------------

    DATA_RAW_DIR: str = "data/raw"
    DATA_PROCESSED_DIR: str = "data/processed"
    MODELS_DIR: str = "models"

    KB_JSON_PATH: str = "data/raw/knowledge_base/knowledge_base_faq.json"
    KB_STORE_DIR: str = "models/kb_store"

    CLASSIFIER_MODEL_DIR: str = "models/classifier"
    LOG_DIR: str = "logs"

    # -----------------------------
    # Bitext dataset
    # -----------------------------

    BITEXT_CSV_PATH: str = (
        "data/raw/bitext/"
        "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )

    # -----------------------------
    # Final 11 classifier categories
    # -----------------------------

    CATEGORIES: List[str] = field(
        default_factory=lambda: [
            "ACCOUNT",
            "CANCEL",
            "CONTACT",
            "DELIVERY",
            "FEEDBACK",
            "INVOICE",
            "ORDER",
            "PAYMENT",
            "REFUND",
            "SHIPPING",
            "SUBSCRIPTION",
        ]
    )

    # -----------------------------
    # Embedding / RAG
    # -----------------------------

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_TOP_K: int = 3
    RAG_MIN_SCORE: float = 0.3

    # -----------------------------
    # LLM
    # -----------------------------

    LLM_PROVIDER: str = os.getenv(
        "LLM_PROVIDER",
        "local"
    )

    LLM_MODEL_NAME: str = os.getenv(
        "LLM_MODEL_NAME",
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    LLM_API_KEY: str = os.getenv(
        "LLM_API_KEY",
        ""
    )

    LLM_TEMPERATURE: float = 0.2

    # -----------------------------
    # Classifier
    # -----------------------------

    CLASSIFIER_BASE_MODEL: str = "distilbert-base-uncased"
    CLASSIFIER_MAX_LEN: int = 128
    CLASSIFIER_EPOCHS: int = 3
    CLASSIFIER_BATCH_SIZE: int = 16
    CLASSIFIER_LR: float = 2e-5

    # -----------------------------
    # Whisper
    # -----------------------------

    WHISPER_MODEL_SIZE: str = "small"

    # -----------------------------
    # Escalation
    # -----------------------------

    ESCALATE_CONFIDENCE_THRESHOLD: float = 0.6

    # -----------------------------
    # Dashboard
    # -----------------------------

    DASHBOARD_TITLE: str = (
        "AutoTriage — Voice Customer Support AI"
    )


config = Config()