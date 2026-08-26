"""Runtime configuration. Every value comes from the environment (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EVALS_DIR = PROJECT_ROOT / "evals"

load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ch_host: str
    ch_port: int
    ch_user: str
    ch_password: str
    ch_database: str
    ch_secure: bool
    embedding_model: str
    embedding_dim: int
    gemini_model: str

    @property
    def has_gemini_key(self) -> bool:
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings(
        ch_host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        ch_port=int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),
        ch_user=os.getenv("CLICKHOUSE_USER", "reel"),
        ch_password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        ch_database=os.getenv("CLICKHOUSE_DATABASE", "subtext"),
        ch_secure=_bool("CLICKHOUSE_SECURE", False),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        gemini_model=os.getenv("SUBTEXT_MODEL", "gemini-2.5-flash"),
    )
