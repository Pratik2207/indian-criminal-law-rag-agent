"""Single source of configuration, read from environment / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent  # repository root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 6144  # prompts peak at ~3.9k tokens; a smaller KV cache keeps more of the model on GPU
    ollama_max_tokens: int = 1024  # answer length cap (real answers are < 600 tokens)

    # Vector store
    qdrant_db_path: str = str(ROOT / "qdrant_db")
    qdrant_url: str = ""  # set e.g. http://localhost:6333 to use a Qdrant server instead of local files
    qdrant_collection_name: str = "indian_law_v2"

    # Models
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # beat bge-reranker-base in evals, ~10x faster

    # Retrieval
    retrieval_mode: str = "hybrid_rerank"  # dense | hybrid | hybrid_rerank
    candidate_k: int = 30
    top_k: int = 6
    # Minimum reranker score of the best hit; below it the answer abstains.
    # -4.0: 2% false abstention on in-scope eval queries, 4/5 out-of-scope abstained (evals/).
    abstain_threshold: float = -4.0

    # Web search (optional)
    serper_api_key: str = ""
    indian_kanoon_api_token: str = ""

    knowledge_dir: str = str(ROOT / "data" / "statutes")


@lru_cache
def get_settings() -> Settings:
    return Settings()
