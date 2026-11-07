"""App settings. Everything can be overridden with an environment variable."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # Folder from Drive ("best": weights + tokenizer + codebert_config.json). NOT in GitHub.
    CODEBERT_DIR = os.environ.get("CODESENTINEL_MODEL_DIR", os.path.join(ROOT, "models", "codebert", "best"))
    # Saved TF-IDF baseline (in GitHub). Used for offline/fallback demo mode.
    BASELINE_DIR = os.environ.get("CODESENTINEL_BASELINE_DIR", os.path.join(ROOT, "models", "baseline_generic"))
    # "auto" = CodeBERT if its folder exists, else the baseline. "codebert" / "baseline" force one.
    BACKEND = os.environ.get("CODESENTINEL_BACKEND", "auto").lower()

    MAX_INPUT_CHARS = int(os.environ.get("CODESENTINEL_MAX_CHARS", "20000"))   # same cap as the baseline config
    MIN_INPUT_CHARS = 20                                                       # same minimum as cleaning step 4
    MAX_REQUEST_BYTES = int(os.environ.get("CODESENTINEL_MAX_BYTES", str(64 * 1024)))

    DATA_DIR = os.path.join(ROOT, "app", "data")
