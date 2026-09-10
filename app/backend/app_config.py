"""
App-only configuration.

These settings are intentionally separate from src.config.Settings.
The src/ configuration controls local experiments/evaluation, while this
configuration controls the deployed Streamlit/FastAPI application.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file before starting the FastAPI application."
    )


# The names exposed to the application/UI remain "llama" and "qwen".
# These are the actual model IDs used by Groq behind the scenes.
#
# Keep these separate from LLAMA_MODEL_NAME / QWEN_MODEL_NAME in src.config.py.
APP_GENERATOR_MODELS: dict[str, str] = {
    "llama": os.getenv(
        "APP_LLAMA_MODEL",
        "openai/gpt-oss-120b",
    ),
    "qwen": os.getenv(
        "APP_QWEN_MODEL",
        "qwen/qwen3.6-27b",
    ),
}


# Application defaults.
APP_DEFAULT_GENERATOR = os.getenv("APP_DEFAULT_GENERATOR", "llama")


def get_app_generator_model(name: str) -> str:
    """Return the Groq model ID for an application generator name."""
    key = name.lower()

    if key not in APP_GENERATOR_MODELS:
        available = ", ".join(sorted(APP_GENERATOR_MODELS))
        raise ValueError(
            f"Unknown app generator '{name}'. Available: {available}"
        )

    return APP_GENERATOR_MODELS[key]
